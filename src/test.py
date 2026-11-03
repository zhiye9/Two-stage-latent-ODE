"""Evaluate trained models
Expected input:
    paths.output_dir/stage1_model.pt and stage2_model.pt  
    paths.metabolite_csv   metabolites data used for training
    paths.static_csv       static features for stage 2

Output (paths.output_dir/test/):
    stage1_predictions.npz   
    stage2_predictions.npz   
    per_timepoint_mse.csv    
    per_group_mse.csv       
    dynamic_metrics/clinical_indices_true.csv, clinical_indices_pred.csv
    dynamic_metrics/agreement.csv             
    dynamic_metrics/cross_group_agreement.json 
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from analysis import (agreement_table, collect_stage1_predictions, collect_stage2_predictions, cross_group_agreement,
                      insulin_dependent_indices, metabolite_indices_table, per_group_mse_table)
from config import build_argument_parser, load_config
from model import load_stage1_checkpoint, load_stage2_checkpoint
from preprocessing import (MetabolitePreprocessor, StaticFeatureProcessor, TrajectoryDataset, _id_sort_key,
                           _worker_init_fn, collate_fn, normalize_subject_id, read_static_csv, wide_to_long)

logger = logging.getLogger("test")

target_metabolites = ["Glucose", "Insulin", "C-peptid", "Total-TG", "Lactate", "Pyruvate", "Citrate", "bOHbutyrate",
                      "Acetoacetate", "Acetate", "Ala", "Ile", "Leu", "Val", "Total BCAA", "Total-FA", "Remnant-C"]


def main() -> None:
    parser = build_argument_parser("Evaluate the trained two-stage model")
    parser.add_argument("--stage1_model", default=None)
    parser.add_argument("--stage2_model", default=None)
    parser.add_argument("--subjects", choices=["test", "all", "file"], default="test")
    parser.add_argument("--subject_file", default=None)
    parser.add_argument("--space", choices=["original", "normalized"], default="original",
                        help="space in which the cross-group agreement score is computed")
    args = parser.parse_args()
    cfg = load_config(args)
    out = os.path.join(cfg.paths.output_dir, "test")
    os.makedirs(os.path.join(out, "dynamic_metrics"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", force=True)
    device = cfg.resolve_device()

    model1, _ = load_stage1_checkpoint(args.stage1_model or os.path.join(cfg.paths.output_dir, "stage1_model.pt"), device)
    model2, predictor, ckpt2 = load_stage2_checkpoint(
        args.stage2_model or os.path.join(cfg.paths.output_dir, "stage2_model.pt"), device)
    pre = MetabolitePreprocessor.from_state_dict(ckpt2["preprocessor"])
    names = list(ckpt2["metabolite_names"])
    gmi = {g: list(v) for g, v in ckpt2["group_metabolite_indices"].items()}
    subject_column = ckpt2.get("subject_column", cfg.data.subject_column)

    wide_df = pd.read_csv(cfg.paths.metabolite_csv)
    wide_df[subject_column] = wide_df[subject_column].map(normalize_subject_id)
    long_df, _ = wide_to_long(wide_df, subject_column)
    if args.subjects == "test":
        ids = [str(s) for s in ckpt2["test_subject_ids"]]
    elif args.subjects == "all":
        ids = long_df["subject_id"].unique().tolist()
    else:
        if not args.subject_file:
            raise ValueError("--subjects file requires --subject_file")
        with open(args.subject_file) as f:
            ids = [normalize_subject_id(line.strip()) for line in f if line.strip()]
    missing = set(ids) - set(long_df["subject_id"])
    if missing:
        raise KeyError(f"{len(missing)} requested subjects are not in the metabolite table, e.g. {sorted(missing)[:5]}")
    ids = sorted(set(ids), key=_id_sort_key)
    logger.info("Evaluating %d subjects (--subjects %s)", len(ids), args.subjects)
    arrays = pre.transform(long_df[long_df["subject_id"].isin(ids)])

    # Stage 1: reconstruction from the full trajectory
    ds1 = TrajectoryDataset(**arrays)
    preds1 = collect_stage1_predictions(model1, DataLoader(
        ds1, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(0), worker_init_fn=_worker_init_fn), device)

    # Stage 2: prediction from static features + X(0)
    static_df = read_static_csv(cfg.paths.static_csv, subject_column)
    static_proc = StaticFeatureProcessor.from_state_dict(ckpt2["static_processor"])
    ds2 = TrajectoryDataset(**arrays, static=static_proc.transform(static_df, ds1.subject_list))
    preds2 = collect_stage2_predictions(model2, predictor, DataLoader(
        ds2, batch_size=cfg.data.batch_size, shuffle=False, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(0), worker_init_fn=_worker_init_fn), device)

    for stage, predictions in (("stage1", preds1), ("stage2", preds2)):
        true_orig = pre.inverse_transform(predictions["true"])
        true_orig[predictions["mask"] == 0] = np.nan
        np.savez(os.path.join(out, f"{stage}_predictions.npz"),
                 true=predictions["true"], pred=predictions["pred"], mask=predictions["mask"],
                 times_h=predictions["times"][0], true_orig=true_orig,
                 pred_orig=pre.inverse_transform(predictions["pred"]), z0=predictions["z0"],
                 subject_ids=predictions["subject_ids"], metabolite_names=np.array(names))

    times_h = preds2["times"][0]
    times_min = times_h * 60.0
    true_orig = pre.inverse_transform(preds2["true"])
    true_orig[preds2["mask"] == 0] = np.nan
    pred_orig = pre.inverse_transform(preds2["pred"])

    # reconstruction performance
    performance = {
        "n_subjects": len(ids), "subjects": args.subjects, "n_metabolites": len(names),
        "persistence": {},
    }
    for key, predictions, masked in (("stage1", preds1, True), ("stage2", preds2, True),
                                      ("stage2_unmasked", preds2, False)):
        valid = np.isfinite(predictions["true"]) & np.isfinite(predictions["pred"])
        if masked:
            valid &= predictions["mask"] > 0
        t, p = predictions["true"][valid], predictions["pred"][valid]
        performance[key] = {"mse": float(mean_squared_error(t, p)) if len(t) else np.nan,
                            "mae": float(mean_absolute_error(t, p)) if len(t) else np.nan, "n": int(valid.sum())}
    persist = np.broadcast_to(preds1["true"][:, :1], preds1["true"].shape)
    for key, sl in (("mse", slice(None)), ("mse_t_gt_0", slice(1, None))):
        t, p = preds1["true"][:, sl], persist[:, sl]
        valid = (preds1["mask"][:, sl] > 0) & np.isfinite(t) & np.isfinite(p)
        performance["persistence"][key] = float(mean_squared_error(t[valid], p[valid])) if valid.any() else np.nan

    timepoint_metrics = {"time_min": [float(t * 60.0) for t in times_h]}
    for stage, predictions in (("stage1", preds1), ("stage2", preds2)):
        values = []
        for i in range(predictions["true"].shape[1]):
            t, p = predictions["true"][:, i], predictions["pred"][:, i]
            valid = (predictions["mask"][:, i] > 0) & np.isfinite(t) & np.isfinite(p)
            values.append(float(mean_squared_error(t[valid], p[valid])) if valid.any() else np.nan)
        timepoint_metrics[f"{stage}_mse"] = values
    pd.DataFrame(timepoint_metrics).to_csv(os.path.join(out, "per_timepoint_mse.csv"), index=False)
    g1 = per_group_mse_table(preds1["true"], preds1["pred"], preds1["mask"], gmi)
    g2 = per_group_mse_table(preds2["true"], preds2["pred"], preds2["mask"], gmi)
    persist = np.repeat(preds1["true"][:, :1], preds1["true"].shape[1], axis=1)
    gp = per_group_mse_table(preds1["true"], persist, preds1["mask"], gmi)
    pd.DataFrame({"group": g1["group"], "n_metabolites": g1["n_metabolites"], "stage1_mse_mean": g1["mse_mean"],
                  "stage2_mse_mean": g2["mse_mean"], "persistence_mse_mean": gp["mse_mean"]}).to_csv(
        os.path.join(out, "per_group_mse.csv"), index=False)
    logger.info("MSE (masked, normalised log space): stage 1 %.4f | stage 2 %.4f | persistence %.4f",
                performance["stage1"]["mse"], performance["stage2"]["mse"], performance["persistence"]["mse"])

    # ---- dynamic phenotype metrics (Stage 2 predictions, original units) ---
    dyn = os.path.join(out, "dynamic_metrics")
    subjects = list(preds2["subject_ids"])
    idx_true = metabolite_indices_table(true_orig, times_min, names, subjects, target_metabolites)
    idx_pred = metabolite_indices_table(pred_orig, times_min, names, subjects, target_metabolites)
    ins_true = insulin_dependent_indices(true_orig, times_min, names, subjects)
    ins_pred = insulin_dependent_indices(pred_orig, times_min, names, subjects)
    pd.concat([ins_true, idx_true], axis=1).to_csv(os.path.join(dyn, "clinical_indices_true.csv"), index_label="subject_id")
    pd.concat([ins_pred, idx_pred], axis=1).to_csv(os.path.join(dyn, "clinical_indices_pred.csv"), index_label="subject_id")
    agreement_table(pd.concat([ins_true, idx_true], axis=1), pd.concat([ins_pred, idx_pred], axis=1)).to_csv(
        os.path.join(dyn, "agreement.csv"), index=False)

    if args.space == "normalized":
        cross = cross_group_agreement(preds2["true"], preds2["pred"], times_min, gmi)
    else:
        cross = cross_group_agreement(true_orig, pred_orig, times_min, gmi)
    cross_out = {k: v for k, v in cross.items() if k not in ("upper_true", "upper_pred")}
    cross_out["space"] = args.space
    with open(os.path.join(dyn, "cross_group_agreement.json"), "w") as f:
        json.dump(cross_out, f, indent=2)
    logger.info("Cross-group agreement (%s space): Spearman rho %.3f over %d group pairs",
                args.space, cross["spearman_rho"], cross["n_pairs"])


if __name__ == "__main__":
    main()
