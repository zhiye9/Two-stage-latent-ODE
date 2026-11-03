"""Configuration for training, evaluation and embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

group_rules = [
    ("Cholesterol", lambda m: m.endswith("-C")),
    ("Triglycerides", lambda m: m.endswith("-TG")),
    ("Phospholipids", lambda m: m.endswith("-PL")),
    ("Cholesteryl_esters", lambda m: m.endswith("-CE")),
    ("Free_cholesterol", lambda m: m.endswith("-FC")),
    ("Total_lipids", lambda m: m.endswith("-L")),
    ("Particles", lambda m: m.endswith("-P")),
    ("Apolipoproteins", lambda m: m.startswith("Apo")),
    ("Fatty_acids", lambda m: m in {"Unsaturation", "Omega-3", "Omega-6", "DHA", "LA", "MUFA", "PUFA", "SFA", "Total-FA"}),
    ("Amino_acids", lambda m: m in {"Ala", "Gln", "Gly", "His", "Ile", "Leu", "Total BCAA", "Tyr", "Val"}),
    ("Glycolysis", lambda m: m in {"Citrate", "Glucose", "Lactate", "Pyruvate", "C-peptid", "GlycA", "Insulin", "GIP", "GLP1", "Glucagon"}),
    ("Ketone_bodies", lambda m: m in {"Acetate", "Acetoacetate", "Acetone", "bOHbutyrate"}),
    ("Other_lipids", lambda m: m in {"Phosphoglyc", "Cholines", "Phosphatidylc", "Sphingomyelins"}),
    ("Fluid_balance", lambda m: m in {"Creatinine", "Albumin"}),
]

group_loss_weights: Dict[str, float] = {"Glycolysis": 10.0, "Ketone_bodies": 10.0, "Amino_acids": 10.0}


# Stage 2 inputs.
static_columns: List[str] = [
    "Sex", "BMI", "Fat_percent", "Fat_mass", "Muscle_mass", "WAIST",
    "Waist-hip-ratio", "Waist-height-ratio", "FITNESS",
]


@dataclass
class PathConfig:
    metabolite_csv: str = "data/metabolites.csv"
    static_csv: str = "data/static.csv"
    output_dir: str = "runs/reference"
    device: str = "cuda"

@dataclass
class DataConfig:
    subject_column: str = "copsacno"
    test_size: float = 0.2
    random_seed: int = 3402
    batch_size: int = 16
    missingness_threshold: float = 0.40
    variance_threshold: float = 1e-5
    log_epsilon: float = 1e-5
    time_divisor: float = 60.0
    static_columns: List[str] = field(default_factory=lambda: list(static_columns))
    categorical_static_columns: List[str] = field(default_factory=lambda: ["Sex"])
    num_workers: int = 0


@dataclass
class Stage1Config:
    """Grouped latent ODE on full trajectories, without static inputs."""
    n_epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler_min_lr: float = 1e-6
    early_stop_patience: int = 100
    log_every: int = 10

    # architecture
    activation: str = "elu"
    ode_activation: str = "tanh"
    d_baseline: int = 96
    baseline_hidden: int = 128
    d_dynamic: int = 96
    gru_ode_hidden: int = 32
    d_fusion: int = 96
    encoder_dropout: float = 0.1
    ode_hidden: int = 296
    ode_n_layers: int = 3
    ode_solver: str = "dopri5"
    ode_rtol: float = 1e-4
    ode_atol: float = 1e-5
    decoder_hidden: int = 96
    decoder_n_layers: int = 2
    dims_per_group: int = 10

    # losses
    beta_start: float = 0.0
    beta_end: float = 1e-4
    kl_anneal_epochs: int = 500
    free_bits: float = 0.1
    recon_delta_weight: float = 1.0
    baseline_weight: float = 1.0
    group_loss_weights: Dict[str, float] = field(default_factory=lambda: dict(group_loss_weights))

    ard_init: float = 1.0
    gate_init: float = -2.0
    lambda_ard: float = 0.5
    lambda_sparsity: float = 0.5
    target_sparsity: float = 0.4
    temperature_init: float = 1.0
    temperature_min: float = 0.1
    temperature_anneal_rate: float = 0.01

    # grouped decoder
    beta_constraint: str = "softplus"
    per_group_beta_init: Dict[str, float] = field(default_factory=lambda: {
        "Cholesterol": 0.2000,
        "Triglycerides": 0.2000,
        "Phospholipids": 0.2000,
        "Cholesteryl_esters": 0.2000,
        "Free_cholesterol": 0.2000,
        "Total_lipids": 0.2000,
        "Particles": 0.2000,
        "Apolipoproteins": 0.2000,
        "Fatty_acids": 0.2000,
        "Other_lipids": 0.2000,
        "Fluid_balance": 0.2009,
        "Amino_acids": 0.4870,
        "Ketone_bodies": 0.7561,
        "Glycolysis": 0.9356,
    })
    beta_init_default: float = 0.4
    soft_connection_init_within: float = 3.0
    soft_connection_init_cross: float = -3.0
    lambda_cross_group: float = 0.75
    cross_group_penalty_type: str = "l2"


@dataclass
class Stage2Config:
    n_epochs: int = 400
    lr_new: float = 1e-3
    lr_finetune: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    early_stop_patience: int = 100
    log_every: int = 10

    activation: str = "elu"
    d_static: int = 96
    d_fusion: int = 128
    dropout: float = 0.1
    film_n_layers: int = 2
    film_hidden_dim: int = 96
    gamma_clamp: List[float] = field(default_factory=lambda: [0.5, 2.0])
    beta_clamp: List[float] = field(default_factory=lambda: [-1.0, 1.0])

    baseline_weight: float = 1.0
    beta_kl: float = 1e-4
    lambda_cross_group: float = 0.05
    cross_group_penalty_type: str = "l2"
    group_loss_weights: Dict[str, float] = field(default_factory=lambda: dict(group_loss_weights))

    train_predictor_components: bool = True


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        for section_name, section in d.items():
            setattr(cfg, section_name, type(getattr(cfg, section_name))(**section))
        return cfg

    def resolve_device(self) -> str:
        if self.paths.device != "auto":
            return self.paths.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="JSON configuration file")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set stage1.n_epochs=2")
    return parser


def load_config(args: argparse.Namespace) -> Config:
    if args.config:
        with open(args.config) as f:
            cfg = Config.from_dict(json.load(f))
    else:
        cfg = Config()
    for item in args.set:
        key, value = item.split("=", 1)
        section_name, key = key.strip().split(".", 1)
        value = value.strip()
        target = getattr(cfg, section_name)
        current = getattr(target, key)
        if isinstance(current, bool):
            value = value.lower() in {"1", "true", "yes", "y"}
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, (list, dict)) or current is None:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        setattr(target, key, value)
    return cfg
