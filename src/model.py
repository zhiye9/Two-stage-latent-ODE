"""Model components of the two-stage grouped latent ODE.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Normal, kl_divergence
from torchdiffeq import odeint

from config import Stage1Config, Stage2Config


# model components
activations = {
    "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "gelu": nn.GELU,
    "silu": nn.SiLU, "tanh": nn.Tanh, "softplus": nn.Softplus,
}


def decoder_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, activation: str) -> nn.Sequential:
    """Linear-activation, then (n_layers - 1) x Linear-LayerNorm-activation, then a linear output layer."""
    layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), activations[activation.lower().strip()]()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), activations[activation.lower().strip()]()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class ODEFunc(nn.Module):
    """dz/dt = f_theta(z): MLP with small Xavier initialisation for stable dynamics."""

    def __init__(self, dim: int, hidden_dim: int, n_layers: int = 2, activation: str = "tanh"):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(dim, hidden_dim), activations[activation.lower().strip()]()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activations[activation.lower().strip()]()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# encoder
class GRUODECell(nn.Module):
    """GRU update at observation times; Euler step of a small ODE between observations."""

    def __init__(self, input_dim: int, hidden_dim: int, ode_hidden: int = 32, ode_activation: str = "tanh"):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim, ode_hidden, n_layers=2, activation=ode_activation)

    def forward(self, x: torch.Tensor, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.max() > 0:
            h = h + delta_t.unsqueeze(-1) * self.ode_func(None, h)
        return self.gru(x, h)


class GRUODEEncoder(nn.Module):
    """Encodes the T0-corrected trajectory (with its mask) backwards in time into h_dynamic."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, ode_hidden: int,
                 activation: str, ode_activation: str):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), activations[activation.lower().strip()](), nn.Linear(hidden_dim, hidden_dim))
        self.cell = GRUODECell(hidden_dim, hidden_dim, ode_hidden, ode_activation)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, times: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, max_len, _ = x.shape
        embedded = self.input_embedding(torch.cat([x * mask, mask], dim=-1))
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        for t in range(max_len - 1, -1, -1):
            delta_t = times[:, t] - times[:, t - 1] if t > 0 else torch.zeros(batch, device=x.device)
            active = (lengths > t).float().unsqueeze(-1)
            h = self.cell(embedded[:, t], h, delta_t) * active + h * (1 - active)
        return self.output(h)


class BaselineEncoder(nn.Module):
    """Fasting metabolite profile X(0) -> h_baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, n_layers: int, dropout: float, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                       activations[activation.lower().strip()](), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.net(x0)


# soft connection
class ARDSoftGating(nn.Module):
    """ARD scaling (1 / sqrt(alpha)) followed by Gumbel-sigmoid gates on a block of latent dimensions."""

    def __init__(self, n_dims: int, ard_init: float = 1.0, gate_init: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((n_dims,), math.log(ard_init)))
        self.gate_logits = nn.Parameter(torch.full((n_dims,), float(gate_init)))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    @property
    def gate_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def set_temperature(self, temperature: float) -> None:
        self.temperature.fill_(float(temperature))

    def gates(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.gate_logits).clamp(1e-8, 1 - 1e-8)
            return torch.sigmoid((self.gate_logits - torch.log(-torch.log(u))) / self.temperature)
        return torch.sigmoid(self.gate_logits)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * (1.0 / torch.sqrt(self.alpha + 1e-8)) * self.gates()

class GroupedARDSoftGating(nn.Module):
    """One ARDSoftGating module per metabolite group, applied to that group's latent block."""

    def __init__(self, group_dim_allocation: Dict[str, List[int]], ard_init: float, gate_init: float, temperature: float):
        super().__init__()
        self.group_dim_allocation = {g: list(v) for g, v in group_dim_allocation.items()}
        self.modules_by_group = nn.ModuleDict()
        for group, dims in self.group_dim_allocation.items():
            self.modules_by_group[group] = ARDSoftGating(len(dims), ard_init, gate_init, temperature)
            self.register_buffer(f"idx_{group}", torch.tensor(dims, dtype=torch.long))

    def set_temperature(self, temperature: float) -> None:
        for module in self.modules_by_group.values():
            module.set_temperature(temperature)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gated = z.clone()
        for group, module in self.modules_by_group.items():
            idx = getattr(self, f"idx_{group}")
            gated[..., idx] = module(z[..., idx])
        return gated

    def compute_total_loss(self, lambda_ard: float, lambda_sparsity: float, target_sparsity: float) -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for module in self.modules_by_group.values():
            alpha = module.alpha
            ard = (0.5 * (alpha - 1 - torch.log(alpha))).sum()
            sparsity = torch.abs(1.0 - module.gate_probs.mean() - target_sparsity)
            total = total + lambda_ard * ard + lambda_sparsity * sparsity
        return total

    def gate_probabilities(self) -> Dict[str, np.ndarray]:
        return {g: m.gate_probs.detach().cpu().numpy() for g, m in self.modules_by_group.items()}


# group decoder
class GroupedDecoder(nn.Module):
    """Per-group MLP decoders whose outputs reach all metabolites through learnable sigmoid connections.

    delta(t) = sum_g sigmoid(l_g) * dec_g(z_g(t)),  X_pred(t) = X0_pred + beta * delta(t)
    with one learnable beta per metabolite (initialised per group) and X0_pred = baseline_decoder(h_baseline).
    """

    def __init__(self, latent_dim: int, baseline_dim: int, n_metabolites: int,
                 group_dim_allocation: Dict[str, List[int]], group_metabolite_indices: Dict[str, List[int]],
                 hidden_dim: int, n_layers: int, activation: str, beta_init: Dict[str, float],
                 beta_constraint: str = "softplus", init_within: float = 3.0, init_cross: float = -3.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_metabolites = n_metabolites
        self.group_dim_allocation = {g: list(v) for g, v in group_dim_allocation.items()}
        self.group_metabolite_indices = {g: list(v) for g, v in group_metabolite_indices.items()}
        self.beta_constraint = beta_constraint
        self.init_within = init_within
        self.init_cross = init_cross

        self.group_decoders = nn.ModuleDict()
        self.connection_logits = nn.ParameterDict()
        self.beta_raw = nn.ParameterDict()
        for group, dims in self.group_dim_allocation.items():
            met_idx = self.group_metabolite_indices[group]
            self.group_decoders[group] = decoder_mlp(len(dims), hidden_dim, n_metabolites, n_layers, activation)
            logits = torch.full((n_metabolites,), float(init_cross))
            logits[met_idx] = float(init_within)
            self.connection_logits[group] = nn.Parameter(logits)
            self.beta_raw[group] = nn.Parameter(torch.full(
                (len(met_idx),), math.log(math.exp(beta_init[group]) - 1 + 1e-8)
                if beta_constraint == "softplus" else float(beta_init[group])))
            self.register_buffer(f"lat_idx_{group}", torch.tensor(dims, dtype=torch.long))
            self.register_buffer(f"met_idx_{group}", torch.tensor(met_idx, dtype=torch.long))
        self.baseline_decoder = decoder_mlp(baseline_dim, hidden_dim, n_metabolites, n_layers, activation)

    @property
    def beta(self) -> torch.Tensor:
        beta = torch.ones(self.n_metabolites, device=next(self.parameters()).device)
        for group in self.group_decoders:
            raw = self.beta_raw[group]
            beta[getattr(self, f"met_idx_{group}")] = functional.softplus(raw) if self.beta_constraint == "softplus" else raw
        return beta

    def forward(self, z: torch.Tensor, h_baseline: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_3d = z.dim() == 3
        z_flat = z.reshape(-1, z.shape[-1]) if is_3d else z
        delta = torch.zeros(z_flat.shape[0], self.n_metabolites, device=z.device)
        for group, decoder in self.group_decoders.items():
            delta = delta + torch.sigmoid(self.connection_logits[group]) * decoder(z_flat[:, getattr(self, f"lat_idx_{group}")])
        x0_pred = self.baseline_decoder(h_baseline)
        beta = self.beta
        if is_3d:
            delta = delta.reshape(z.shape[0], z.shape[1], -1)
            x_pred = x0_pred.unsqueeze(1) + beta * delta
        else:
            x_pred = x0_pred + beta * delta
        return x_pred, delta, x0_pred

    def compute_cross_group_penalty(self, lambda_cross: float, penalty_type: str = "l2") -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for group in self.group_decoders:
            weights = torch.sigmoid(self.connection_logits[group])
            cross_mask = torch.ones_like(weights)
            cross_mask[getattr(self, f"met_idx_{group}")] = 0.0
            cross = weights * cross_mask
            total = total + lambda_cross * (cross.abs().sum() if penalty_type == "l1" else cross.pow(2).sum())
        return total

    def connection_summary(self) -> List[Dict]:
        rows = []
        for group in self.group_decoders:
            weights = torch.sigmoid(self.connection_logits[group]).detach().cpu()
            within = torch.zeros_like(weights)
            within[getattr(self, f"met_idx_{group}").cpu()] = 1.0
            n_within = int(within.sum())
            n_cross = int(weights.numel() - n_within)
            rows.append({
                "group": group,
                "within_mean": float((weights * within).sum() / max(n_within, 1)),
                "cross_mean": float((weights * (1 - within)).sum() / max(n_cross, 1)),
                "cross_max": float((weights * (1 - within)).max()),
                "n_within": n_within, "n_cross": n_cross,
            })
        return rows

    def beta_summary(self) -> Dict[str, float]:
        return {g: float((functional.softplus(self.beta_raw[g]) if self.beta_constraint == "softplus"
                          else self.beta_raw[g]).mean()) for g in self.group_decoders}


# stage 1
class GroupedLatentODE(nn.Module):
    def __init__(self, n_metabolites: int, group_dim_allocation: Dict[str, List[int]],
                 group_metabolite_indices: Dict[str, List[int]], cfg: Stage1Config, beta_init: Dict[str, float]):
        super().__init__()
        self.cfg = cfg
        self.n_metabolites = n_metabolites
        self.group_dim_allocation = {g: list(v) for g, v in group_dim_allocation.items()}
        self.group_metabolite_indices = {g: list(v) for g, v in group_metabolite_indices.items()}
        self.latent_dim = sum(len(v) for v in group_dim_allocation.values())

        self.baseline_encoder = BaselineEncoder(n_metabolites, cfg.baseline_hidden, cfg.d_baseline, 2,
                                                cfg.encoder_dropout, cfg.activation)
        self.dynamic_encoder = GRUODEEncoder(n_metabolites, cfg.d_dynamic, cfg.d_dynamic, cfg.gru_ode_hidden,
                                             cfg.activation, cfg.ode_activation)
        self.fusion_alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.baseline_proj = nn.Linear(cfg.d_baseline, cfg.d_fusion) if cfg.d_baseline != cfg.d_fusion else None
        self.dynamic_proj = nn.Linear(cfg.d_dynamic, cfg.d_fusion) if cfg.d_dynamic != cfg.d_fusion else None
        self.posterior_mu = nn.Linear(cfg.d_fusion, self.latent_dim)
        self.posterior_logvar = nn.Linear(cfg.d_fusion, self.latent_dim)
        for layer in (self.posterior_mu, self.posterior_logvar):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.posterior_logvar.bias, -2.0)

        self.ode_func = ODEFunc(self.latent_dim, cfg.ode_hidden, cfg.ode_n_layers, cfg.ode_activation)
        self.gating = GroupedARDSoftGating(group_dim_allocation, cfg.ard_init, cfg.gate_init, cfg.temperature_init)
        self.grouped_decoder = GroupedDecoder(
            self.latent_dim, cfg.d_baseline, n_metabolites, group_dim_allocation, group_metabolite_indices,
            cfg.decoder_hidden, cfg.decoder_n_layers, cfg.activation, beta_init, cfg.beta_constraint,
            cfg.soft_connection_init_within, cfg.soft_connection_init_cross)

    @property
    def fusion_alpha(self) -> float:
        return float(torch.sigmoid(self.fusion_alpha_logit))

    def encode(self, metabolites: torch.Tensor, mask: torch.Tensor, times: torch.Tensor,
               lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Posterior parameters of z0 and the baseline embedding."""
        x0 = metabolites[:, 0, :]
        h_baseline = self.baseline_encoder(x0)
        h_dynamic = self.dynamic_encoder(metabolites - x0.unsqueeze(1), mask, times, lengths)
        alpha = torch.sigmoid(self.fusion_alpha_logit)
        h = alpha * (self.baseline_proj(h_baseline) if self.baseline_proj is not None else h_baseline)
        h = h + (1 - alpha) * (self.dynamic_proj(h_dynamic) if self.dynamic_proj is not None else h_dynamic)
        return self.posterior_mu(h), self.posterior_logvar(h), h_baseline

    @staticmethod
    def gather_times(times: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sorted unique observation times of the batch and, per (subject, position), the index into them."""
        valid = torch.arange(times.shape[1], device=times.device).unsqueeze(0) < lengths.unsqueeze(1)
        unique_times = torch.unique(times[valid], sorted=True)
        return unique_times, torch.searchsorted(unique_times, times.contiguous()).clamp(max=len(unique_times) - 1)

    def solve_ode(self, z0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Integrate dz/dt = f_theta(z) from z0 over the (sorted) times; returns (batch, n_times, latent)."""
        cfg = self.cfg
        if cfg.ode_solver == "dopri5":
            z = odeint(self.ode_func, z0, times, method="dopri5", rtol=cfg.ode_rtol, atol=cfg.ode_atol)
        else:
            z = odeint(self.ode_func, z0, times, method=cfg.ode_solver)
        return z.permute(1, 0, 2)

    def forward(self, metabolites: torch.Tensor, mask: torch.Tensor, times: torch.Tensor,
                lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar, h_baseline = self.encode(metabolites, mask, times, lengths)
        if self.training:
            z0 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z0 = mu
        z0 = self.gating(z0)
        unique_times, index = self.gather_times(times, lengths)
        z_traj = self.solve_ode(z0, unique_times)
        recon, delta_recon, baseline_recon = self.grouped_decoder(
            self.gating(z_traj[torch.arange(z_traj.shape[0], device=z_traj.device).unsqueeze(1), index]), h_baseline)
        return {
            "recon": recon, "delta_recon": delta_recon, "baseline_recon": baseline_recon,
            "posterior_mu": mu, "posterior_logvar": logvar, "z0": z0, "z_traj": z_traj,
            "unique_times": unique_times, "h_baseline": h_baseline,
        }


# stage 2
class StaticEncoder(nn.Module):
    """Static features (with their missingness mask) -> h_static."""

    def __init__(self, n_static: int, d_hidden: int, d_output: int, dropout: float, n_layers: int, activation: str):
        super().__init__()
        self.d_output = d_output
        if n_static == 0:
            self.net = None
            return
        layers: List[nn.Module] = []
        in_dim = n_static
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, d_hidden), nn.LayerNorm(d_hidden), activations[activation.lower().strip()](), nn.Dropout(dropout)]
            in_dim = d_hidden
        layers += [nn.Linear(in_dim, d_output), activations[activation.lower().strip()]()]
        self.net = nn.Sequential(*layers)

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros(static.shape[0], self.d_output, device=static.device)
        return self.net(static)


class FiLMGenerator(nn.Module):
    """condition -> (gamma, beta); initialised near the identity transformation (gamma = 1, beta = 0)."""

    def __init__(self, condition_dim: int, output_dim: int, hidden_dim: int, activation: str,
                 gamma_clamp: Sequence[float], beta_clamp: Sequence[float]):
        super().__init__()
        self.output_dim = output_dim
        self.gamma_clamp = tuple(gamma_clamp)
        self.beta_clamp = tuple(beta_clamp)
        self.generator = nn.Sequential(nn.Linear(condition_dim, hidden_dim), activations[activation.lower().strip()](),
                                       nn.Linear(hidden_dim, 2 * output_dim))
        final = self.generator[-1]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            final.bias[:output_dim] = 1.0
            final.bias[output_dim:] = 0.0

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = torch.chunk(self.generator(condition), 2, dim=-1)
        return torch.clamp(gamma, *self.gamma_clamp), torch.clamp(beta, *self.beta_clamp)


class FiLMLayer(nn.Module):
    """h = dropout(act(gamma * LayerNorm(Linear(x)) + beta))."""

    def __init__(self, in_dim: int, out_dim: int, condition_dim: int, film_hidden_dim: int, dropout: float,
                 activation: str, gamma_clamp: Sequence[float], beta_clamp: Sequence[float]):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.activation = activations[activation.lower().strip()]()
        self.dropout = nn.Dropout(dropout)
        self.film = FiLMGenerator(condition_dim, out_dim, film_hidden_dim, activation, gamma_clamp, beta_clamp)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(condition)
        return self.dropout(self.activation(gamma * self.layer_norm(self.linear(x)) + beta))


class FiLMFusionNetwork(nn.Module):
    """FiLM-modulated fusion of [h_static, h_baseline] (modulation generated from h_static)."""

    def __init__(self, d_static: int, d_baseline: int, d_fusion: int, dropout: float, n_layers: int,
                 film_hidden_dim: int, activation: str, gamma_clamp: Sequence[float], beta_clamp: Sequence[float]):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = d_static + d_baseline
        for _ in range(n_layers):
            self.layers.append(FiLMLayer(in_dim, d_fusion, d_static, film_hidden_dim, dropout, activation, gamma_clamp, beta_clamp))
            in_dim = d_fusion
        self.output_layer = nn.Sequential(nn.Linear(d_fusion, d_fusion), activations[activation.lower().strip()]())

    def forward(self, h_static: torch.Tensor, h_baseline: torch.Tensor) -> torch.Tensor:
        h = torch.cat([h_static, h_baseline], dim=-1)
        for layer in self.layers:
            h = layer(h, h_static)
        return self.output_layer(h)

    def film_parameters(self, h_static: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [layer.film(h_static) for layer in self.layers]


class FiLMZ0Predictor(nn.Module):
    """Predicts z0 ~ N(mu, sigma^2) from static features and the fasting profile X(0)."""

    def __init__(self, n_static: int, baseline_encoder: nn.Module, d_baseline: int, latent_dim: int, cfg: Stage2Config):
        super().__init__()
        self.latent_dim = latent_dim
        self.static_encoder = StaticEncoder(n_static, cfg.d_static, cfg.d_static, cfg.dropout, 2, cfg.activation)
        self.baseline_encoder = copy.deepcopy(baseline_encoder)
        self.fusion_net = FiLMFusionNetwork(cfg.d_static, d_baseline, cfg.d_fusion, cfg.dropout, cfg.film_n_layers,
                                            cfg.film_hidden_dim, cfg.activation, cfg.gamma_clamp, cfg.beta_clamp)
        self.z0_mu = nn.Linear(cfg.d_fusion, latent_dim)
        self.z0_logvar = nn.Linear(cfg.d_fusion, latent_dim)
        for layer in (self.z0_mu, self.z0_logvar):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.z0_logvar.bias, -2.0)

    def forward(self, static: torch.Tensor, x0: torch.Tensor):
        h_static = self.static_encoder(static)
        h_baseline = self.baseline_encoder(x0)
        h = self.fusion_net(h_static, h_baseline)
        mu = self.z0_mu(h)
        logvar = torch.clamp(self.z0_logvar(h), min=-10.0, max=2.0)
        z0 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return z0, mu, logvar, h_baseline

    def get_param_groups(self, lr_new: float, lr_finetune: float, train_predictor_components: bool = True) -> List[Dict]:
        """Parameter groups with differential learning rates (materialised lists, never generators)."""
        groups = [{"params": list(self.z0_mu.parameters()) + list(self.z0_logvar.parameters()), "lr": lr_new, "name": "z0_heads"}]
        if train_predictor_components:
            groups = [
                {"params": list(self.static_encoder.parameters()), "lr": lr_new, "name": "static_encoder"},
                {"params": list(self.fusion_net.parameters()), "lr": lr_new, "name": "fusion_net"},
            ] + groups + [
                {"params": list(self.baseline_encoder.parameters()), "lr": lr_finetune, "name": "baseline_encoder"},
            ]
        return [g for g in groups if len(g["params"]) > 0]


def stage2_forward(model: GroupedLatentODE, predictor: FiLMZ0Predictor, static: torch.Tensor, x0: torch.Tensor,
                   times: torch.Tensor, lengths: torch.Tensor):
    """z0 from static + X(0) -> frozen ODE -> gating -> grouped decoder. Returns X_pred, delta, X0_pred, mu, logvar."""
    z0, mu, logvar, h_baseline = predictor(static, x0)
    unique_times, index = model.gather_times(times, lengths)
    z_traj = model.solve_ode(z0, unique_times)
    x_pred, delta, x0_pred = model.grouped_decoder(
        model.gating(z_traj[torch.arange(z_traj.shape[0], device=z_traj.device).unsqueeze(1), index]), h_baseline)
    return x_pred, delta, x0_pred, mu, logvar


# loss function
def stage1_loss(model: GroupedLatentODE, batch: Dict, epoch: int, cfg: Stage1Config,
                feature_weights: Optional[torch.Tensor], device: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    metabolites = batch["metabolites"].to(device)
    mask = batch["mask"].to(device)
    out = model(metabolites, mask, batch["times"].to(device), batch["lengths"].to(device))

    x0_true = metabolites[:, 0, :]
    weighted_mask = mask if feature_weights is None else mask * feature_weights.view(1, 1, -1)
    denom = weighted_mask.sum().clamp(min=1.0)
    delta_error = functional.mse_loss(model.grouped_decoder.beta * out["delta_recon"],
                             metabolites - x0_true.unsqueeze(1), reduction="none")
    baseline_error = functional.mse_loss(out["baseline_recon"], x0_true, reduction="none")
    full_error = functional.mse_loss(out["recon"], metabolites, reduction="none")
    recon_delta = (delta_error * weighted_mask).sum() / denom
    recon_baseline = (baseline_error.mean() if feature_weights is None else
                      (baseline_error * feature_weights.view(1, -1)).sum() / (feature_weights.sum() * metabolites.shape[0]))
    recon_full = (full_error * weighted_mask).sum() / denom
    mse_unweighted = (full_error * mask).sum() / mask.sum().clamp(min=1.0)
    mu, logvar = out["posterior_mu"], out["posterior_logvar"]
    kl = kl_divergence(Normal(mu, torch.exp(0.5 * logvar)), Normal(torch.zeros_like(mu), torch.ones_like(mu)))
    if cfg.free_bits > 0:
        kl = kl.clamp(min=cfg.free_bits)
    kl = kl.mean(dim=0).sum()
    if epoch >= cfg.kl_anneal_epochs:
        beta_kl = cfg.beta_end
    else:
        beta_kl = cfg.beta_start + epoch / cfg.kl_anneal_epochs * (cfg.beta_end - cfg.beta_start)
    ard = model.gating.compute_total_loss(cfg.lambda_ard, cfg.lambda_sparsity, cfg.target_sparsity)
    cross = model.grouped_decoder.compute_cross_group_penalty(cfg.lambda_cross_group, cfg.cross_group_penalty_type)

    total = (cfg.recon_delta_weight * recon_delta + cfg.baseline_weight * recon_baseline
             + beta_kl * kl + ard + cross)
    return total, {
        "loss": total.item(), "recon_delta": recon_delta.item(), "recon_baseline": recon_baseline.item(),
        "recon_full": recon_full.item(), "mse_unweighted": mse_unweighted.item(), "kl": kl.item(),
        "kl_weight": beta_kl, "ard": ard.item(), "cross_group": cross.item(),
    }


def stage2_loss(model: GroupedLatentODE, predictor: FiLMZ0Predictor, batch: Dict, cfg: Stage2Config,
                feature_weights: Optional[torch.Tensor], device: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    metabolites = batch["metabolites"].to(device)
    mask = batch["mask"].to(device)
    x0 = metabolites[:, 0, :]
    x_pred, delta_pred, x0_pred, mu, logvar = stage2_forward(
        model, predictor, batch["static"].to(device), x0, batch["times"].to(device), batch["lengths"].to(device))

    weighted_mask = mask if feature_weights is None else mask * feature_weights.view(1, 1, -1)
    denom = weighted_mask.sum().clamp(min=1.0)
    full_error = functional.mse_loss(x_pred, metabolites, reduction="none")
    delta_error = functional.mse_loss(delta_pred, metabolites - x0.unsqueeze(1), reduction="none")
    baseline_error = functional.mse_loss(x0_pred, x0, reduction="none")
    recon = (full_error * weighted_mask).sum() / denom
    mse_unweighted = (full_error * mask).sum() / mask.sum().clamp(min=1.0)
    recon_delta = (delta_error * weighted_mask).sum() / denom
    baseline = (baseline_error.mean() if feature_weights is None else
                (baseline_error * feature_weights.view(1, -1)).sum() / (feature_weights.sum() * metabolites.shape[0]))
    kl = kl_divergence(Normal(mu, torch.exp(0.5 * logvar)), Normal(torch.zeros_like(mu), torch.ones_like(mu))).mean(dim=0).sum()
    cross = model.grouped_decoder.compute_cross_group_penalty(cfg.lambda_cross_group, cfg.cross_group_penalty_type)
    total = recon + cfg.baseline_weight * baseline + cfg.beta_kl * kl + cross
    return total, {
        "loss": total.item(), "recon_full": recon.item(), "mse_unweighted": mse_unweighted.item(),
        "recon_delta": recon_delta.item(), "recon_baseline": baseline.item(), "kl": kl.item(),
        "cross_group": cross.item(),
    }


# checkpoints
def load_stage1_checkpoint(path: str, device: str):
    """Rebuild the Stage 1 model from ``stage1_model.pt`` written by train.py. Returns (model, checkpoint)."""
    from config import Config
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:   
        ckpt = torch.load(path, map_location=device)
    cfg = Config.from_dict(ckpt["config"])
    model = GroupedLatentODE(ckpt["n_metabolites"], ckpt["group_dim_allocation"], ckpt["group_metabolite_indices"],
                             cfg.stage1, ckpt["beta_init"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_stage2_checkpoint(path: str, device: str):
    """Rebuild model + predictor from ``stage2_model.pt``. Returns (model, predictor, checkpoint)."""
    model, ckpt = load_stage1_checkpoint(path, device)
    predictor = FiLMZ0Predictor(ckpt["n_static"], model.baseline_encoder, model.cfg.d_baseline, model.latent_dim,
                                Stage2Config(**ckpt["config"].get("stage2", {}))).to(device)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.eval()
    predictor.eval()
    return model, predictor, ckpt
