"""Train the two-stage latent ODE
Expected input:
    metabolite_csv  one row per subject. metabolite time-seris
    static_csv      one row per subject: static features (sex, BMI, Ffat_percent, fat_mass, muscle_mass, waist, whr,
                    waist-height-ratio, FITNESS).

Output (written to paths.output_dir):
    stage1_model.pt               
    stage2_model.pt                      
    per_group_mse.csv             test MSE per metabolite group
    per_timepoint_mse.csv         test MSE per time point 
    beta_per_group.csv            learnable beta per group (initial, after Stage 1, after Stage 2)
    gate_probabilities.csv        ARD gate probability of every latent dimension
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from analysis import collect_stage1_predictions, collect_stage2_predictions, per_group_mse_table
from config import Config, build_argument_parser, load_config
from model import FiLMZ0Predictor, GroupedLatentODE, stage1_loss, stage2_loss
from preprocessing import (MetabolitePreprocessor, StaticFeatureProcessor, TrajectoryDataset, _id_sort_key,
                           _worker_init_fn, assign_groups, collate_fn, normalize_subject_id, read_static_csv, wide_to_long)

logger = logging.getLogger("train")


# ultilies
def run_training_loop(name: str, n_epochs: int, train_epoch: Callable[[int], Dict[str, float]],
                      evaluate: Callable[[int], Dict[str, float]], after_epoch: Callable[[Dict[str, float]], None],
                      snapshot: Callable[[], Dict], patience: int, log_every: int, selection_metric: str = "recon_full"):
    """Generic epoch loop with early stopping on the test selection metric. Returns (history, best_state, best_epoch, best_value)."""
    history: List[Dict[str, float]] = []
    best_value, best_epoch, best_state, waited = float("inf"), -1, None, 0
    start = time.time()
    for epoch in range(n_epochs):
        train_metrics = train_epoch(epoch)
        val_metrics = evaluate(epoch)
        after_epoch(val_metrics)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"test_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if val_metrics[selection_metric] < best_value:
            best_value, best_epoch, best_state, waited = val_metrics[selection_metric], epoch, snapshot(), 0
        else:
            waited += 1
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info("%s epoch %4d | train loss %.5f recon_full %.5f | test recon_full %.5f (best %.5f @ %d) | %.0fs",
                        name, epoch, train_metrics["loss"], train_metrics["recon_full"], val_metrics["recon_full"],
                        best_value, best_epoch, time.time() - start)
        if waited >= patience:
            logger.info("%s: early stopping at epoch %d (no improvement for %d epochs)", name, epoch, patience)
            break
    logger.info("%s finished in %.1f min; best test %s = %.6f at epoch %d", name, (time.time() - start) / 60,
                selection_metric, best_value, best_epoch)
    return history, best_state, best_epoch, best_value


def main() -> None:
    cfg = load_config(build_argument_parser("Train Stage 1 and Stage 2 of the grouped latent ODE").parse_args())
    out = cfg.paths.output_dir
    os.makedirs(out, exist_ok=True)
    device = cfg.resolve_device()
    random.seed(cfg.data.random_seed)
    np.random.seed(cfg.data.random_seed)
    torch.manual_seed(cfg.data.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.data.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Device: %s | output: %s", device, out)

    wide_df = pd.read_csv(cfg.paths.metabolite_csv)
    wide_df[cfg.data.subject_column] = wide_df[cfg.data.subject_column].map(normalize_subject_id)
    long_df, metabolite_names = wide_to_long(wide_df, cfg.data.subject_column)
    all_ids = long_df["subject_id"].unique().tolist()
    train_ids, test_ids = train_test_split(sorted(set(all_ids), key=_id_sort_key),
                                          test_size=cfg.data.test_size, random_state=cfg.data.random_seed)
    logger.info("Subjects: %d total, %d train, %d test", len(all_ids), len(train_ids), len(test_ids))
    train_rows = long_df[long_df["subject_id"].isin(train_ids)]

    pre = MetabolitePreprocessor(cfg.data).fit(train_rows, metabolite_names)
    names = pre.valid_metabolites
    group_names, group_metabolite_indices = assign_groups(names)
    dims_per_group = cfg.stage1.dims_per_group
    group_dim_allocation = {g: list(range(i * dims_per_group, (i + 1) * dims_per_group))
                            for i, g in enumerate(group_names)}
    latent_dim = len(group_names) * dims_per_group
    logger.info("%d metabolites in %d groups -> latent dimension %d", len(names), len(group_names), latent_dim)
    data = {
        "long_df": long_df, "metabolite_names": names, "preprocessor": pre,
        "train_ids": train_ids, "test_ids": test_ids, "train": pre.transform(train_rows),
        "test": pre.transform(long_df[long_df["subject_id"].isin(test_ids)]),
        "group_names": group_names, "group_metabolite_indices": group_metabolite_indices,
        "group_dim_allocation": group_dim_allocation, "latent_dim": latent_dim,
    }

    # Stage 1
    s1 = cfg.stage1
    # prepare datasets
    train_ds1, test_ds1 = TrajectoryDataset(**data["train"]), TrajectoryDataset(**data["test"])
    train_loader = DataLoader(train_ds1, batch_size=cfg.data.batch_size, shuffle=True, collate_fn=collate_fn,
                               generator=torch.Generator().manual_seed(cfg.data.random_seed),
                               worker_init_fn=_worker_init_fn, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds1, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn,
                              generator=torch.Generator().manual_seed(cfg.data.random_seed),
                              worker_init_fn=_worker_init_fn, num_workers=cfg.data.num_workers)
    beta_init = {g: float(s1.per_group_beta_init.get(g, s1.beta_init_default)) for g in group_dim_allocation}
    model = GroupedLatentODE(len(names), group_dim_allocation, group_metabolite_indices, s1, beta_init).to(device)
    weights = torch.ones(len(names), dtype=torch.float32, device=device)
    for group, weight in s1.group_loss_weights.items():
        if group in group_metabolite_indices:
            weights[group_metabolite_indices[group]] = float(weight)
        else:
            logger.warning("Loss weight for unknown group %r ignored", group)
    logger.info("Stage 1 model: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=s1.learning_rate, weight_decay=s1.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=s1.n_epochs, eta_min=s1.scheduler_min_lr)

    def train_epoch(epoch: int) -> Dict[str, float]:
        model.train()
        model.gating.set_temperature(max(
            s1.temperature_init * math.exp(-s1.temperature_anneal_rate * epoch), s1.temperature_min))
        batches = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, components = stage1_loss(model, batch, epoch, s1, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), s1.grad_clip)
            optimizer.step()
            batches.append(components)
        return pd.DataFrame(batches).mean(skipna=False).to_dict()

    @torch.no_grad()
    def evaluate(epoch: int) -> Dict[str, float]:
        model.eval()
        return pd.DataFrame([stage1_loss(model, batch, epoch, s1, weights, device)[1]
                             for batch in test_loader]).mean(skipna=False).to_dict()

    hist1, best_state, best_epoch1, best_val1 = run_training_loop(
        "Stage 1", s1.n_epochs, train_epoch, evaluate, lambda _: scheduler.step(),
        lambda: copy.deepcopy(model.state_dict()), s1.early_stop_patience, s1.log_every)
    model.load_state_dict(best_state)
    model.eval()

    # checkpoints
    checkpoint = {
        "config": asdict(cfg),
        "n_metabolites": len(data["metabolite_names"]),
        "metabolite_names": list(data["metabolite_names"]),
        "group_names": list(data["group_names"]),
        "group_metabolite_indices": {g: list(v) for g, v in data["group_metabolite_indices"].items()},
        "group_dim_allocation": {g: list(v) for g, v in data["group_dim_allocation"].items()},
        "latent_dim": int(data["latent_dim"]),
        "beta_init": {g: float(cfg.stage1.per_group_beta_init.get(g, cfg.stage1.beta_init_default)) for g in data["group_names"]},
        "preprocessor": data["preprocessor"].state_dict(),
        "subject_column": cfg.data.subject_column,
        "train_subject_ids": list(data["train_ids"]),
        "test_subject_ids": list(data["test_ids"]),
    }
    ckpt1 = {**checkpoint, "model_state_dict": model.state_dict(), "epoch": int(best_epoch1), "best_metric": float(best_val1)}
    torch.save(ckpt1, os.path.join(out, "stage1_model.pt"))
    logger.info("Saved stage1_model.pt")

    # Stage 1 performance
    preds1 = {split: collect_stage1_predictions(model, DataLoader(
                  ds, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn,
                  generator=torch.Generator().manual_seed(cfg.data.random_seed), worker_init_fn=_worker_init_fn), device)
              for split, ds in (("train", train_ds1), ("test", test_ds1))}
    beta_after_stage1 = model.grouped_decoder.beta_summary()
    connections = [dict(stage="stage1", **row) for row in model.grouped_decoder.connection_summary()]

    # Stage 2
    s2 = cfg.stage2
    static_df = read_static_csv(cfg.paths.static_csv, cfg.data.subject_column)
    static_proc = StaticFeatureProcessor(cfg.data.static_columns, cfg.data.categorical_static_columns)
    static_proc.fit(static_df, train_ds1.subject_list)
    train_ds2 = TrajectoryDataset(**data["train"], static=static_proc.transform(static_df, train_ds1.subject_list))
    test_ds2 = TrajectoryDataset(**data["test"], static=static_proc.transform(static_df, test_ds1.subject_list))
    train_loader = DataLoader(train_ds2, batch_size=cfg.data.batch_size, shuffle=True, collate_fn=collate_fn,
                               generator=torch.Generator().manual_seed(cfg.data.random_seed),
                               worker_init_fn=_worker_init_fn, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(test_ds2, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn,
                              generator=torch.Generator().manual_seed(cfg.data.random_seed),
                              worker_init_fn=_worker_init_fn, num_workers=cfg.data.num_workers)

    predictor = FiLMZ0Predictor(static_proc.n_combined, model.baseline_encoder,
                                model.cfg.d_baseline, model.latent_dim, s2).to(device)
    
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.grouped_decoder.parameters():
        p.requires_grad_(True)
    weights = torch.ones(len(names), dtype=torch.float32, device=device)
    for group, weight in s2.group_loss_weights.items():
        if group in group_metabolite_indices:
            weights[group_metabolite_indices[group]] = float(weight)
        else:
            logger.warning("Loss weight for unknown group %r ignored", group)
    param_groups = predictor.get_param_groups(s2.lr_new, s2.lr_finetune, s2.train_predictor_components)
    param_groups.append({"params": list(model.grouped_decoder.parameters()), "lr": s2.lr_finetune, "name": "grouped_decoder"})
    for g in param_groups:
        logger.info("Stage 2 parameter group %-18s %10s params, lr=%g", g["name"], f"{sum(p.numel() for p in g['params']):,}", g["lr"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=s2.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=s2.scheduler_factor, patience=s2.scheduler_patience,
                                  min_lr=s2.scheduler_min_lr)

    def train_epoch(epoch: int) -> Dict[str, float]:
        model.train()
        predictor.train()
        batches = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, components = stage2_loss(model, predictor, batch, s2, weights, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), s2.grad_clip)
            torch.nn.utils.clip_grad_norm_(model.grouped_decoder.parameters(), s2.grad_clip)
            optimizer.step()
            batches.append(components)
        return pd.DataFrame(batches).mean(skipna=False).to_dict()

    @torch.no_grad()
    def evaluate(epoch: int) -> Dict[str, float]:
        model.eval()
        predictor.eval()
        return pd.DataFrame([stage2_loss(model, predictor, batch, s2, weights, device)[1]
                             for batch in test_loader]).mean(skipna=False).to_dict()

    hist2, best_state, best_epoch2, best_val2 = run_training_loop(
        "Stage 2", s2.n_epochs, train_epoch, evaluate, lambda val: scheduler.step(val["loss"]),
        lambda: {"model": copy.deepcopy(model.state_dict()), "predictor": copy.deepcopy(predictor.state_dict())},
        s2.early_stop_patience, s2.log_every)
    model.load_state_dict(best_state["model"])
    predictor.load_state_dict(best_state["predictor"])
    model.eval()
    predictor.eval()
    ckpt2 = {
        **checkpoint,
        "model_state_dict": model.state_dict(), "predictor_state_dict": predictor.state_dict(),
        "n_static": int(static_proc.n_combined), "static_processor": static_proc.state_dict(),
        "static_feature_names": static_proc.feature_names, "epoch": int(best_epoch2), "best_metric": float(best_val2),
    }
    torch.save(ckpt2, os.path.join(out, "stage2_model.pt"))
    logger.info("Saved stage2_model.pt")

    preds2 = {split: collect_stage2_predictions(
        model, predictor, DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn,
                                      generator=torch.Generator().manual_seed(cfg.data.random_seed),
                                      worker_init_fn=_worker_init_fn), device)
        for split, ds in (("train", train_ds2), ("test", test_ds2))}
    connections += [dict(stage="stage2", **row) for row in model.grouped_decoder.connection_summary()]

    # performance 
    gmi = data["group_metabolite_indices"]
    times_h = np.unique(preds1["test"]["times"])
    performance = {
        "n_subjects": {"train": len(data["train_ids"]), "test": len(data["test_ids"])},
        "n_metabolites": len(data["metabolite_names"]), "latent_dim": data["latent_dim"],
        "time_points_min": [float(t * 60) for t in times_h], "device": device,
        "stage1": {"best_epoch": int(best_epoch1), "best_test_weighted_recon_full": float(best_val1),
                   "n_parameters": sum(p.numel() for p in model.parameters())},
        "stage2": {"best_epoch": int(best_epoch2), "best_test_weighted_recon_full": float(best_val2),
                   "n_parameters_predictor": sum(p.numel() for p in predictor.parameters()),
                   "train_predictor_components": cfg.stage2.train_predictor_components},
        "persistence": {},
    }
    for split in ("train", "test"):
        p1, p2 = preds1[split], preds2[split]
        for stage, key, predictions, masked in (("stage1", split, p1, True), ("stage2", split, p2, True),
                                                 ("stage2", f"{split}_unmasked", p2, False)):
            valid = np.isfinite(predictions["true"]) & np.isfinite(predictions["pred"])
            if masked:
                valid &= predictions["mask"] > 0
            t, p = predictions["true"][valid], predictions["pred"][valid]
            performance[stage][key] = {"mse": float(mean_squared_error(t, p)) if len(t) else np.nan,
                                        "mae": float(mean_absolute_error(t, p)) if len(t) else np.nan, "n": int(valid.sum())}
        persist = np.broadcast_to(p1["true"][:, :1], p1["true"].shape)
        performance["persistence"][split] = {}
        for key, sl in (("mse", slice(None)), ("mse_t_gt_0", slice(1, None))):
            t, p = p1["true"][:, sl], persist[:, sl]
            valid = (p1["mask"][:, sl] > 0) & np.isfinite(t) & np.isfinite(p)
            performance["persistence"][split][key] = float(mean_squared_error(t[valid], p[valid])) if valid.any() else np.nan

    timepoint_metrics = {"time_min": [float(t * 60.0) for t in times_h]}
    for stage, predictions in (("stage1", preds1["test"]), ("stage2", preds2["test"])):
        values = []
        for i in range(predictions["true"].shape[1]):
            t, p = predictions["true"][:, i], predictions["pred"][:, i]
            valid = (predictions["mask"][:, i] > 0) & np.isfinite(t) & np.isfinite(p)
            values.append(float(mean_squared_error(t[valid], p[valid])) if valid.any() else np.nan)
        timepoint_metrics[f"{stage}_test_mse"] = values
    pd.DataFrame(timepoint_metrics).to_csv(os.path.join(out, "per_timepoint_mse.csv"), index=False)

    g1 = per_group_mse_table(preds1["test"]["true"], preds1["test"]["pred"], preds1["test"]["mask"], gmi)
    g2 = per_group_mse_table(preds2["test"]["true"], preds2["test"]["pred"], preds2["test"]["mask"], gmi)
    persist = np.repeat(preds1["test"]["true"][:, :1], preds1["test"]["true"].shape[1], axis=1)
    gp = per_group_mse_table(preds1["test"]["true"], persist, preds1["test"]["mask"], gmi)
    pd.DataFrame({"group": g1["group"], "n_metabolites": g1["n_metabolites"],
                  "stage1_test_mse_mean": g1["mse_mean"], "stage1_test_mse_median": g1["mse_median"],
                  "stage2_test_mse_mean": g2["mse_mean"], "stage2_test_mse_median": g2["mse_median"],
                  "persistence_test_mse_mean": gp["mse_mean"]}).to_csv(os.path.join(out, "per_group_mse.csv"), index=False)

    beta_after_stage2 = model.grouped_decoder.beta_summary()
    pd.DataFrame([{"group": g, "beta_init": ckpt1["beta_init"][g], "beta_after_stage1": beta_after_stage1[g],
                   "beta_after_stage2": beta_after_stage2[g]} for g in data["group_names"]]).to_csv(
        os.path.join(out, "beta_per_group.csv"), index=False)
    gate_rows = []

    logger.info("Test MSE (masked, normalised log space): stage 1 %.4f | stage 2 %.4f | persistence %.4f",
                performance["stage1"]["test"]["mse"], performance["stage2"]["test"]["mse"],
                performance["persistence"]["test"]["mse"])
    logger.info("Done. Outputs in %s", out)


if __name__ == "__main__":
    main()
