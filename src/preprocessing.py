"""Data preprocessing for the two-stage grouped latent ODE.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import GROUP_RULES as group_rules, DataConfig

logger = logging.getLogger("preprocessing")

def normalize_subject_id(value) -> str:
    """Represent a subject id as a string ('3', '3.0' and 3 all become '3')."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip()
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def _id_sort_key(value: str):
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


# transform the table 
def wide_to_long(wide_df: pd.DataFrame, subject_column: str) -> Tuple[pd.DataFrame, List[str]]:
    value_columns = [c for c in wide_df.columns if c != subject_column]
    names, times = [], []
    for col in value_columns:
        name, time_str = col.rsplit("_", 1)
        times.append(float(time_str))
        names.append(name)
    metabolite_names = sorted(set(names))

    long = wide_df.melt(id_vars=[subject_column], value_vars=value_columns, var_name="column", value_name="value")
    column_map = dict(zip(value_columns, zip(names, times)))
    long["metabolite"] = long["column"].map(lambda c: column_map[c][0])
    long["time"] = long["column"].map(lambda c: column_map[c][1])
    table = long.pivot(index=[subject_column, "time"], columns="metabolite", values="value")
    table = table.reindex(columns=metabolite_names)
    table = table.reset_index().rename(columns={subject_column: "subject_id"})
    table.columns.name = None
    table["subject_id"] = table["subject_id"].astype(str)
    table = table.sort_values(["subject_id", "time"], key=lambda s: s.map(_id_sort_key) if s.name == "subject_id" else s)
    table = table.reset_index(drop=True)
    logger.info("wide -> long: %d subjects, %d time points, %d metabolites",
                table["subject_id"].nunique(), table["time"].nunique(), len(metabolite_names))
    return table, metabolite_names


# metabolite preprocessor
class MetabolitePreprocessor:
    """Filtering, log transform and z-scoring fitted on the training subjects."""

    def __init__(self, cfg: DataConfig):
        self.missingness_threshold = cfg.missingness_threshold
        self.variance_threshold = cfg.variance_threshold
        self.log_epsilon = cfg.log_epsilon
        self.time_divisor = cfg.time_divisor
        self.valid_metabolites: List[str] = []
        self.dropped_metabolites: Dict[str, str] = {}
        self.train_mean: Optional[np.ndarray] = None
        self.train_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ transforms
    def log_transform(self, data: np.ndarray) -> np.ndarray:
        return np.log(np.clip(data.astype(np.float64), a_min=-self.log_epsilon + 1e-12, a_max=None) + self.log_epsilon)

    # ------------------------------------------------------------------ fit / transform
    def fit(self, long_df: pd.DataFrame, metabolite_names: Sequence[str]) -> "MetabolitePreprocessor":
        data = long_df[list(metabolite_names)]
        missingness = data.isnull().mean()
        variance = data.var(skipna=True)
        keep = (missingness <= self.missingness_threshold) & (variance > self.variance_threshold)
        self.valid_metabolites = [m for m in metabolite_names if bool(keep[m])]
        self.dropped_metabolites = {}
        for m in metabolite_names:
            if not keep[m]:
                reason = []
                if missingness[m] > self.missingness_threshold:
                    reason.append(f"missingness {missingness[m]:.2f}")
                if not variance[m] > self.variance_threshold:
                    reason.append(f"variance {variance[m]:.3g}")
                self.dropped_metabolites[m] = ", ".join(reason)
        logger.info("Metabolite filtering: %d -> %d (dropped: %s)", len(metabolite_names),
                    len(self.valid_metabolites), ", ".join(self.dropped_metabolites) or "none")
        return self.fit_statistics(long_df, self.valid_metabolites)

    def fit_statistics(self, long_df: pd.DataFrame, metabolite_names: Sequence[str]) -> "MetabolitePreprocessor":
        """Recompute the log mean / std on ``long_df`` for a fixed metabolite list (no filtering)."""
        self.valid_metabolites = list(metabolite_names)
        log_data = self.log_transform(long_df[self.valid_metabolites].values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.train_mean = np.nanmean(log_data, axis=0)
            self.train_std = np.nanstd(log_data, axis=0)
        self.train_std = np.where(self.train_std < 1e-8, 1.0, self.train_std)
        return self

    def transform(self, long_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Returns normalised values (N_rows, D), mask, times in hours and subject ids (row-aligned)."""
        raw = long_df[self.valid_metabolites].values.astype(np.float64)
        normalized = (self.log_transform(raw) - self.train_mean) / self.train_std
        return {
            "values": np.nan_to_num(normalized, nan=0.0).astype(np.float32),
            "mask": (~np.isnan(raw)).astype(np.float32),
            "times": (long_df["time"].values / self.time_divisor).astype(np.float32),
            "subject_ids": long_df["subject_id"].astype(str).values,
        }

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Normalised log space -> original concentration units."""
        return np.exp(normalized * self.train_std + self.train_mean) - self.log_epsilon

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> Dict:
        return {
            "missingness_threshold": self.missingness_threshold,
            "variance_threshold": self.variance_threshold,
            "log_epsilon": self.log_epsilon,
            "time_divisor": self.time_divisor,
            "valid_metabolites": list(self.valid_metabolites),
            "dropped_metabolites": dict(self.dropped_metabolites),
            "train_mean": [float(x) for x in self.train_mean],
            "train_std": [float(x) for x in self.train_std],
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "MetabolitePreprocessor":
        obj = cls(DataConfig(missingness_threshold=state["missingness_threshold"],
                             variance_threshold=state["variance_threshold"],
                             log_epsilon=state["log_epsilon"], time_divisor=state["time_divisor"]))
        obj.valid_metabolites = list(state["valid_metabolites"])
        obj.dropped_metabolites = dict(state.get("dropped_metabolites", {}))
        obj.train_mean = np.array(state["train_mean"], dtype=np.float64)
        obj.train_std = np.array(state["train_std"], dtype=np.float64)
        return obj


# =============================================================================
# Static features
# =============================================================================
class StaticFeatureProcessor:
    """Standardised continuous features + one-hot categorical features, each with a mask."""

    def __init__(self, columns: Sequence[str], categorical_columns: Sequence[str]):
        self.requested_columns = list(columns)
        self.categorical_requested = list(categorical_columns)
        self.continuous_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.categories: Dict[str, List[str]] = {}

    @property
    def n_raw(self) -> int:
        return len(self.continuous_columns) + sum(len(v) for v in self.categories.values())

    @property
    def n_combined(self) -> int:
        return 2 * self.n_raw

    @property
    def feature_names(self) -> List[str]:
        names = list(self.continuous_columns)
        for col in self.categorical_columns:
            names.extend(f"{col}={c}" for c in self.categories[col])
        return names

    def fit(self, df: pd.DataFrame, subject_ids: Sequence[str]) -> "StaticFeatureProcessor":
        available = [c for c in self.requested_columns if c in df.columns]
        missing = [c for c in self.requested_columns if c not in df.columns]
        if missing:
            logger.warning("Static columns not found and ignored: %s", missing)
        if not available:
            raise ValueError("None of the requested static columns is present")
        self.categorical_columns = [c for c in available if c in self.categorical_requested]
        self.continuous_columns = [c for c in available if c not in self.categorical_requested]
        rows = df.loc[list(subject_ids)]
        if self.continuous_columns:
            cont = rows[self.continuous_columns].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
            cont = np.nan_to_num(cont, nan=0.0)
            self.scaler_mean = cont.mean(axis=0)
            scale = cont.std(axis=0)
            self.scaler_scale = np.where(scale < 1e-12, 1.0, scale)
        for col in self.categorical_columns:
            self.categories[col] = sorted(rows[col].map(_categorical_value).unique().tolist())
        logger.info("Static features: %d continuous, %d categorical -> %d values + %d mask entries",
                    len(self.continuous_columns), len(self.categorical_columns), self.n_raw, self.n_raw)
        return self

    def transform(self, df: pd.DataFrame, subject_ids: Sequence[str]) -> np.ndarray:
        """Returns (n_subjects, 2 * n_raw) float32 array ``[features, mask]`` in the order of subject_ids."""
        rows = df.loc[list(subject_ids)]
        features, masks = [], []
        if self.continuous_columns:
            cont = rows[self.continuous_columns].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
            masks.append((~np.isnan(cont)).astype(np.float32))
            cont = np.nan_to_num(cont, nan=0.0)
            features.append(((cont - self.scaler_mean) / self.scaler_scale).astype(np.float32))
        for col in self.categorical_columns:
            values = rows[col].map(_categorical_value).values
            onehot = pd.get_dummies(pd.Categorical(values, categories=self.categories[col]), dtype=np.float32).to_numpy()
            features.append(onehot)
            masks.append(np.ones_like(onehot))
        return np.hstack(features + masks).astype(np.float32)

    def state_dict(self) -> Dict:
        return {
            "requested_columns": self.requested_columns,
            "categorical_requested": self.categorical_requested,
            "continuous_columns": self.continuous_columns,
            "categorical_columns": self.categorical_columns,
            "scaler_mean": None if self.scaler_mean is None else [float(x) for x in self.scaler_mean],
            "scaler_scale": None if self.scaler_scale is None else [float(x) for x in self.scaler_scale],
            "categories": {k: list(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "StaticFeatureProcessor":
        obj = cls(state["requested_columns"], state["categorical_requested"])
        obj.continuous_columns = list(state["continuous_columns"])
        obj.categorical_columns = list(state["categorical_columns"])
        obj.scaler_mean = None if state["scaler_mean"] is None else np.array(state["scaler_mean"])
        obj.scaler_scale = None if state["scaler_scale"] is None else np.array(state["scaler_scale"])
        obj.categories = {k: list(v) for k, v in state["categories"].items()}
        return obj


def _categorical_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    return normalize_subject_id(value)


def read_static_csv(path: str, subject_column: str) -> pd.DataFrame:
    """Static / phenotype table indexed by the normalised subject id."""
    df = pd.read_csv(path)
    candidates = [c for c in df.columns if c.lower() == subject_column.lower()]
    if not candidates:
        raise KeyError(f"Subject column {subject_column!r} not found in {path}")
    col = candidates[0]
    df[col] = df[col].map(normalize_subject_id)
    df = df.drop_duplicates(subset=col, keep="first").set_index(col)
    df.index.name = subject_column
    return df


# dataset
class TrajectoryDataset(Dataset):
    """One item per subject: trajectory tensors sorted by time, plus optional static features."""

    def __init__(self, values: np.ndarray, mask: np.ndarray, times: np.ndarray, subject_ids: np.ndarray,
                 static: Optional[np.ndarray] = None, static_subject_ids: Optional[Sequence[str]] = None):
        self.subject_list = sorted({str(s) for s in subject_ids}, key=_id_sort_key)
        self.n_metabolites = values.shape[1]
        subject_ids = np.asarray([str(s) for s in subject_ids])
        self.items: Dict[str, Dict[str, torch.Tensor]] = {}
        for subj in self.subject_list:
            rows = np.where(subject_ids == subj)[0]
            order = rows[np.argsort(times[rows], kind="stable")]
            self.items[subj] = {
                "metabolites": torch.tensor(values[order], dtype=torch.float32),
                "mask": torch.tensor(mask[order], dtype=torch.float32),
                "times": torch.tensor(times[order], dtype=torch.float32),
            }
        if static is None:
            self.n_static = 0
            self.static = {subj: torch.zeros(0) for subj in self.subject_list}
        else:
            if static_subject_ids is None:
                static_subject_ids = self.subject_list
            lookup = {str(s): i for i, s in enumerate(static_subject_ids)}
            self.n_static = static.shape[1]
            self.static = {subj: torch.tensor(static[lookup[subj]], dtype=torch.float32) for subj in self.subject_list}

    def __len__(self) -> int:
        return len(self.subject_list)

    def __getitem__(self, idx: int) -> Dict:
        subj = self.subject_list[idx]
        item = dict(self.items[subj])
        item["static"] = self.static[subj]
        item["subject_id"] = subj
        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length trajectories; returns lengths and subject ids as well."""
    metabolites = pad_sequence([item["metabolites"] for item in batch], batch_first=True)
    mask = pad_sequence([item["mask"] for item in batch], batch_first=True)
    times = pad_sequence([item["times"] for item in batch], batch_first=True)
    static = torch.stack([item["static"] for item in batch])
    lengths = torch.tensor([item["metabolites"].shape[0] for item in batch], dtype=torch.long)
    subject_ids = [item["subject_id"] for item in batch]
    return {"metabolites": metabolites, "mask": mask, "times": times, "static": static,
            "lengths": lengths, "subject_ids": subject_ids}


def _worker_init_fn(worker_id: int) -> None:
    import random
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def assign_groups(metabolite_names: Sequence[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """Map metabolites to groups with GROUP_RULES (rule order = group order).

    Returns the non-empty group names in rule order and ``{group: [metabolite indices]}``.
    Raises if a metabolite matches no rule or more than one rule.
    """
    indices: Dict[str, List[int]] = {name: [] for name, _ in group_rules}
    for idx, met in enumerate(metabolite_names):
        matches = [name for name, rule in group_rules if rule(met)]
        if len(matches) != 1:
            raise ValueError(f"Metabolite {met!r} matches {len(matches)} group rules: {matches}")
        indices[matches[0]].append(idx)
    group_names = [name for name, _ in group_rules if indices[name]]
    for name in group_names:
        logger.info("Group %-20s %3d metabolites", name, len(indices[name]))
    return group_names, {name: indices[name] for name in group_names}


def to_3d(values: np.ndarray, mask: np.ndarray, times: np.ndarray, subject_ids: np.ndarray):
    """Row-wise arrays -> (n_subjects, n_times, n_metabolites) arrays in sorted subject order.

    Requires every subject to have the same set of time points (true for the diet challenge data).
    Returns values_3d, mask_3d, unique_times (hours) and the subject order.
    """
    subject_ids = np.asarray([str(s) for s in subject_ids])
    subjects = sorted(set(subject_ids), key=_id_sort_key)
    unique_times = np.unique(times)
    values_3d = np.zeros((len(subjects), len(unique_times), values.shape[1]), dtype=np.float32)
    mask_3d = np.zeros_like(values_3d)
    time_index = {float(t): i for i, t in enumerate(unique_times)}
    for s_idx, subj in enumerate(subjects):
        for r in np.where(subject_ids == subj)[0]:
            values_3d[s_idx, time_index[float(times[r])]] = values[r]
            mask_3d[s_idx, time_index[float(times[r])]] = mask[r]
    return values_3d, mask_3d, unique_times, subjects
