"""Evaluation utilities: reconstruction metrics, dynamic phenotype indices, cross-group agreement,
z0 x phenotype correlations and the corresponding figures.

All trajectory arrays are ``(n_subjects, n_times, n_metabolites)``; masks have the same shape
(1 = observed). ``times_min`` is a 1-D array of time points in minutes.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.integrate import trapezoid
from scipy.spatial.distance import cosine
from scipy.stats import false_discovery_control, linregress, pearsonr, spearmanr, variation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger("analysis")


# reconstruct trjectories from the models
@torch.no_grad()
def collect_stage1_predictions(model, loader, device: str) -> Dict[str, np.ndarray]:
    """Reconstructions of the Stage 1 model (posterior mean, deterministic gates)."""
    model.eval()
    true, pred, mask, times, ids, z0 = [], [], [], [], [], []
    for batch in loader:
        metabolites = batch["metabolites"].to(device)
        out = model(metabolites, batch["mask"].to(device), batch["times"].to(device), batch["lengths"].to(device))
        true.append(metabolites.cpu().numpy())
        pred.append(out["recon"].cpu().numpy())
        mask.append(batch["mask"].numpy())
        times.append(batch["times"].numpy())
        z0.append(out["posterior_mu"].cpu().numpy())
        ids.extend(batch["subject_ids"])
    return {"true": np.concatenate(true), "pred": np.concatenate(pred), "mask": np.concatenate(mask),
            "times": np.concatenate(times),
            "z0": np.concatenate(z0), "subject_ids": np.array(ids)}


@torch.no_grad()
def collect_stage2_predictions(model, predictor, loader, device: str) -> Dict[str, np.ndarray]:
    """Trajectories predicted from static features + X(0) only."""
    from model import stage2_forward
    model.eval()
    predictor.eval()
    true, pred, mask, times, ids, z0 = [], [], [], [], [], []
    for batch in loader:
        metabolites = batch["metabolites"].to(device)
        x_pred, _, _, mu, _ = stage2_forward(model, predictor, batch["static"].to(device), metabolites[:, 0, :],
                                             batch["times"].to(device), batch["lengths"].to(device))
        true.append(metabolites.cpu().numpy())
        pred.append(x_pred.cpu().numpy())
        mask.append(batch["mask"].numpy())
        times.append(batch["times"].numpy())
        z0.append(mu.cpu().numpy())
        ids.extend(batch["subject_ids"])
    return {"true": np.concatenate(true), "pred": np.concatenate(pred), "mask": np.concatenate(mask),
            "times": np.concatenate(times),
            "z0": np.concatenate(z0), "subject_ids": np.array(ids)}


# reconstruction errors
def per_group_mse_table(true: np.ndarray, pred: np.ndarray, mask: np.ndarray,
                        group_metabolite_indices: Dict[str, List[int]], exclude_t0: bool = True) -> pd.DataFrame:
    sl = slice(1, None) if exclude_t0 else slice(None)
    rows = []
    for group, idx in group_metabolite_indices.items():
        t, p, m = true[:, sl][:, :, idx], pred[:, sl][:, :, idx], mask[:, sl][:, :, idx] > 0
        counts = m.sum(axis=(1, 2))
        squared_error = np.where(m, (t - p) ** 2, 0.0)
        values = np.divide(squared_error.sum(axis=(1, 2)), counts,
                           out=np.full(true.shape[0], np.nan), where=counts > 0)
        v = values[np.isfinite(values)]
        rows.append({"group": group, "n_metabolites": len(group_metabolite_indices[group]),
                     "mse_mean": float(v.mean()) if len(v) else np.nan,
                     "mse_median": float(np.median(v)) if len(v) else np.nan,
                     "mse_std": float(v.std()) if len(v) else np.nan, "n_subjects": int(len(v))})
    return pd.DataFrame(rows)


# dynamic phenotype indices
def trajectory_indices(times_min: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    """Summary indices of one metabolite trajectory (concentration units, time in minutes)."""
    if np.any(~np.isfinite(values)):
        return {}
    baseline = values[0]
    peak_idx, nadir_idx = int(np.argmax(values)), int(np.argmin(values))
    rise = values[peak_idx] - baseline
    suppression = baseline - values[nadir_idx]
    out = {
        "AUC": float(trapezoid(values, times_min)),
        "iAUC": float(trapezoid(np.maximum(values - baseline, 0), times_min)),
        "dAUC": float(trapezoid(np.maximum(baseline - values, 0), times_min)),
        "net_change": float(values[-1] - baseline),
    }
    mean_val = np.mean(values)
    out["CV"] = float(abs(variation(values, ddof=0))) if abs(mean_val) > 1e-10 else np.nan
    if rise > 1e-10:
        out["Cmax"] = float(values[peak_idx])
        out["Tmax"] = float(times_min[peak_idx])
        out["peak_magnitude"] = float(rise)
        out["recovery_fraction"] = float(1 - abs(values[-1] - baseline) / abs(rise))
        out["clearance_rate"] = (
            float(linregress(times_min[peak_idx:], values[peak_idx:]).slope)
            if peak_idx < len(times_min) - 1 and np.std(values[peak_idx:]) >= 1e-10 else np.nan)
    if suppression > 1e-10:
        out["suppression_depth"] = float(suppression)
        out["suppression_fraction"] = float(suppression / abs(baseline)) if baseline != 0 else np.nan
        out["T_nadir"] = float(times_min[nadir_idx])
        out["suppression_iAUC"] = out["dAUC"]
        out["rebound_magnitude"] = float(values[-1] - values[nadir_idx])
    return out


def metabolite_indices_table(data: np.ndarray, times_min: np.ndarray, met_names: Sequence[str],
                             subject_ids: Sequence[str], metabolites: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per subject; columns ``<metabolite>|<index>``."""
    selected = list(metabolites) if metabolites is not None else list(met_names)
    name_to_idx = {m: i for i, m in enumerate(met_names)}
    records = []
    for s in range(data.shape[0]):
        row = {}
        for met in selected:
            if met not in name_to_idx:
                continue
            for key, value in trajectory_indices(times_min, data[s, :, name_to_idx[met]]).items():
                row[f"{met}|{key}"] = value
        records.append(row)
    return pd.DataFrame(records, index=list(subject_ids))


def insulin_dependent_indices(data: np.ndarray, times_min: np.ndarray, met_names: Sequence[str],
                              subject_ids: Sequence[str]) -> pd.DataFrame:
    """Composite glucose/insulin indices (Matsuda, HOMA-IR, insulinogenic, disposition, HIRI, MISI, iAUC, Cmax)."""
    if "Glucose" not in met_names or "Insulin" not in met_names:
        logger.warning("Glucose and/or Insulin not available: insulin-dependent indices skipped")
        return pd.DataFrame(index=list(subject_ids))
    g_idx, i_idx = list(met_names).index("Glucose"), list(met_names).index("Insulin")
    records = []
    for s in range(data.shape[0]):
        g, ins = data[s, :, g_idx], data[s, :, i_idx]
        row: Dict[str, float] = {}
        if np.all(np.isfinite(g)) and np.all(np.isfinite(ins)):
            g0, i0 = g[0], ins[0]
            denominator = np.sqrt(g0 * i0 * np.mean(g) * np.mean(ins)) if g0 * i0 * np.mean(g) * np.mean(ins) > 0 else 0.0
            row["Matsuda_Index"] = 10000.0 / denominator if denominator > 0 else np.nan
            row["HOMA_IR"] = (g0 * i0) / 22.5 if i0 > 0 else np.nan
            t30 = np.isclose(times_min, 30.0, atol=0.6)
            if np.any(t30):
                dg = g[t30][0] - g0
                row["Insulinogenic_Index"] = (ins[t30][0] - i0) / dg if abs(dg) > 1e-10 else np.nan
                m, igi = row["Matsuda_Index"], row["Insulinogenic_Index"]
                row["Disposition_Index"] = m * igi if np.isfinite(m) and np.isfinite(igi) else np.nan
            row["Insulin_AUC"] = float(trapezoid(ins, times_min))
            row["Insulin_iAUC"] = float(trapezoid(np.maximum(ins - i0, 0), times_min))
            row["Insulin_Cmax"] = float(np.max(ins))
            row["Insulin_Tmax"] = float(times_min[np.argmax(ins)])
            row["Glucose_AUC"] = float(trapezoid(g, times_min))
            row["Glucose_iAUC"] = float(trapezoid(np.maximum(g - g0, 0), times_min))
            row["Glucose_Cmax"] = float(np.max(g))
            row["Glucose_Tmax"] = float(times_min[np.argmax(g)])
            early = times_min <= 30.0 + 1e-6
            if early.sum() >= 2:
                row["HIRI"] = float(trapezoid(np.maximum(g[early] - g0, 0), times_min[early])
                                    * trapezoid(np.maximum(ins[early] - i0, 0), times_min[early]))
            peak = int(np.argmax(g))
            if peak < len(times_min) - 1:
                decline = (g[-1] - g[peak]) / (times_min[-1] - times_min[peak])
                mean_ins = np.mean(ins[peak:])
                row["MISI"] = abs(decline) / mean_ins if mean_ins > 0 else np.nan
            row["Insulin_Glucose_Tmax_delay"] = row["Insulin_Tmax"] - row["Glucose_Tmax"]
        records.append(row)
    return pd.DataFrame(records, index=list(subject_ids))

# cross-group agreement

def cross_group_agreement(true: np.ndarray, pred: np.ndarray, times_min: np.ndarray,
                          group_metabolite_indices: Dict[str, List[int]]) -> Dict:
    """Agreement between the true and predicted cross-group correlation structure.

    For each subject and group: s = median over the group's metabolites of the positive iAUC.
    R_true / R_pred: Pearson correlation of these scores across subjects (groups x groups).
    Score: Spearman rho between the upper triangles of R_true and R_pred (plus Pearson r and RV).
    """
    iauc_t = trapezoid(np.maximum(true - true[:, :1, :], 0.0), x=times_min, axis=1)
    iauc_p = trapezoid(np.maximum(pred - pred[:, :1, :], 0.0), x=times_min, axis=1)
    groups = list(group_metabolite_indices)
    s_t = np.column_stack([np.nanmedian(iauc_t[:, idx], axis=1) for idx in group_metabolite_indices.values()])
    s_p = np.column_stack([np.nanmedian(iauc_p[:, idx], axis=1) for idx in group_metabolite_indices.values()])
    r_t = pd.DataFrame(s_t, columns=groups).corr(method="pearson").to_numpy()
    r_p = pd.DataFrame(s_p, columns=groups).corr(method="pearson").to_numpy()
    matrix_valid = ~(np.isnan(r_p) | np.isnan(r_t))
    a = np.where(matrix_valid, r_p, 0.0).ravel()
    b = np.where(matrix_valid, r_t, 0.0).ravel()
    iu = np.triu_indices(len(groups), k=1)
    upper_t, upper_p = r_t[iu], r_p[iu]
    valid = np.isfinite(upper_t) & np.isfinite(upper_p)
    result = {"group_names": groups, "n_pairs": int(valid.sum()), "n_subjects": int(true.shape[0]),
              "R_true": r_t.tolist(), "R_pred": r_p.tolist(),
              "upper_true": upper_t[valid], "upper_pred": upper_p[valid],
              "spearman_rho": np.nan, "spearman_p": np.nan, "pearson_r": np.nan, "pearson_p": np.nan,
              "rv_coefficient": float(1.0 - cosine(a, b)) if np.any(a) and np.any(b) else np.nan}
    if valid.sum() >= 3:
        rho, p_rho = spearmanr(upper_t[valid], upper_p[valid])
        r, p_r = pearsonr(upper_t[valid], upper_p[valid])
        result.update({"spearman_rho": float(rho), "spearman_p": float(p_rho), "pearson_r": float(r), "pearson_p": float(p_r)})
    return result

def agreement_table(df_true: pd.DataFrame, df_pred: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """Spearman / Pearson / CCC between true and predicted values of every index (column)."""
    rows = []
    for col in df_true.columns:
        if col not in df_pred.columns:
            continue
        t = pd.to_numeric(df_true[col], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(df_pred[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(t) & np.isfinite(p)
        row = {"metric": col, "n": int(valid.sum()), "spearman_rho": np.nan, "spearman_p": np.nan,
               "pearson_r": np.nan, "pearson_p": np.nan, "ccc": np.nan}
        if valid.sum() >= min_n and np.std(t[valid]) > 0 and np.std(p[valid]) > 0:
            t, p = t[valid], p[valid]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p_rho = spearmanr(t, p)
                r, p_r = pearsonr(t, p)
            denom = t.var() + p.var() + (t.mean() - p.mean()) ** 2
            row.update({"spearman_rho": float(rho), "spearman_p": float(p_rho), "pearson_r": float(r),
                        "pearson_p": float(p_r),
                        "ccc": float(2 * np.mean((t - t.mean()) * (p - p.mean())) / denom)
                        if len(t) >= 3 and denom > 1e-12 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)




# z0 and phenotype correlations
def phenotype_correlations(z0: np.ndarray, subject_ids: Sequence[str], phenotypes: pd.DataFrame,
                           phenotype_columns: Sequence[str], group_dim_allocation: Dict[str, List[int]],
                           min_valid: int = 10, alpha: float = 0.05) -> pd.DataFrame:
    """Spearman correlation of every z0 dimension with every phenotype; BH correction over all tests.

    ``phenotypes`` is indexed by the normalised subject id; subjects without phenotypes are dropped.
    """
    ids = [str(s) for s in subject_ids]
    common = [s for s in ids if s in phenotypes.index]
    logger.info("Phenotype correlation: %d of %d subjects have phenotype rows", len(common), len(ids))
    pos = {s: i for i, s in enumerate(ids)}
    z = z0[[pos[s] for s in common]]
    pheno = phenotypes.loc[common]
    dim_to_group = {}
    for group, dims in group_dim_allocation.items():
        for local, d in enumerate(dims):
            dim_to_group[d] = (group, local)
    rows = []
    for col in phenotype_columns:
        if col not in pheno.columns:
            continue
        values = pd.to_numeric(pheno[col], errors="coerce").to_numpy(dtype=float)
        for d in range(z.shape[1]):
            group, local = dim_to_group.get(d, ("", d))
            valid = np.isfinite(values) & np.isfinite(z[:, d])
            row = {"z0_dimension": d, "group": group, "local_dim": local, "phenotype": col,
                   "spearman_rho": np.nan, "p_value": np.nan, "n_valid": int(valid.sum())}
            if valid.sum() >= min_valid and np.std(values[valid]) > 0 and np.std(z[valid, d]) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, p = spearmanr(z[valid, d], values[valid])
                if np.isfinite(rho):
                    row["spearman_rho"], row["p_value"] = float(rho), float(p)
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        p_values = df["p_value"].to_numpy(dtype=float)
        valid = np.isfinite(p_values)
        adjusted = np.full_like(p_values, np.nan)
        if valid.any():
            adjusted[valid] = false_discovery_control(p_values[valid], method="bh")
        df["p_value_fdr"] = adjusted
        df["significant"] = df["p_value_fdr"] < alpha
    return df

