#!/usr/bin/env python3
"""
Two-stage DCT coarse + conditional diffusion refiner (FREQUENCY DOMAIN diffusion) with v-pred.

Drop-in replacement for `FiHard.py`, keeping the same public API:

- CardConfig
- CardForecaster with:
    - .coarse(history) -> (B, T_out, N, 3)
    - .diffusion_loss(history, future_gt) -> (loss, coarse_future)
    - .predict(history, deterministic=True, seed=0) -> refined (B, T_out, N, 3)
    - .forward(history): returns coarse in train mode, refined in eval mode

Stage A (coarse): predicts low-frequency DCT content and can expose future-DCT coefficients directly.
Stage B (refiner): conditional diffusion over future-DCT residuals.
Diffusion is trained and sampled directly in coefficient space:
    x0 = DCT(y_gt) - DCT(y_coarse)

Denoiser predicts v (v-pred):
    x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
    v   = sqrt(ab) * eps - sqrt(1-ab) * x0
This avoids the unstable division by sqrt(ab) used by epsilon-pred when computing x0 at high noise.

This file assumes your project already provides:
- models.simlpe_dct.SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.simlpe_dct import SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix

try:
    import xformers.ops as _xops  # type: ignore
except Exception:
    _xops = None


# ----------------------------- Small utilities --------------------------------

def _assert_shape(t: torch.Tensor, shape: Tuple[Optional[int], ...], name: str) -> None:
    if t.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dims, got {t.ndim} with shape {tuple(t.shape)}")
    for i, s in enumerate(shape):
        if s is not None and int(t.shape[i]) != int(s):
            raise ValueError(f"{name} dim {i} expected {s}, got {t.shape[i]}")

def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device, dtype=emb.dtype)], dim=1)
    return emb

def _make_ddim_timesteps(T: int, steps: int, device: torch.device) -> torch.Tensor:
    """
    Make a strictly decreasing integer timestep schedule with no duplicates.
    Ensures inclusion of T-1 and 0.
    """
    if steps <= 1:
        return torch.tensor([T - 1], device=device, dtype=torch.long)
    base = torch.linspace(0, T - 1, steps, device=device)
    idx = torch.round(base).long()
    idx = torch.cat([idx, torch.tensor([0, T - 1], device=device, dtype=torch.long)], dim=0)
    idx = torch.unique(idx, sorted=True)
    idx = torch.flip(idx, dims=[0])  # descending
    return idx


def _clamp_optional(x: torch.Tensor, min_value: Optional[float], max_value: Optional[float]) -> torch.Tensor:
    if min_value is None and max_value is None:
        return x
    return torch.clamp(x, min=min_value, max=max_value)


def _multihead_attention_fast(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    fast_attn_env = os.environ.get("DIFFHF_USE_FAST_ATTN", "1").strip().lower()
    if fast_attn_env in {"0", "false", "no", "off"}:
        return attn(query, key, value, need_weights=False)
    if not attn.batch_first:
        return attn(query, key, value, need_weights=False)
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        return attn(query, key, value, need_weights=False)

    batch_size, tgt_len, embed_dim = query.shape
    _, src_len, _ = key.shape
    num_heads = int(attn.num_heads)
    head_dim = embed_dim // num_heads
    dropout_p = float(attn.dropout) if attn.training else 0.0

    q_proj, k_proj, v_proj = F._in_projection_packed(
        query,
        key,
        value,
        attn.in_proj_weight,
        attn.in_proj_bias,
    )
    q_proj = q_proj.reshape(batch_size, tgt_len, num_heads, head_dim)
    k_proj = k_proj.reshape(batch_size, src_len, num_heads, head_dim)
    v_proj = v_proj.reshape(batch_size, src_len, num_heads, head_dim)

    if _xops is not None and query.is_cuda:
        try:
            attn_out = _xops.memory_efficient_attention(
                q_proj,
                k_proj,
                v_proj,
                p=dropout_p,
            )
            attn_out = attn_out.reshape(batch_size, tgt_len, embed_dim)
            attn_out = attn.out_proj(attn_out)
            return attn_out, None
        except Exception:
            pass

    q_proj = q_proj.permute(0, 2, 1, 3)
    k_proj = k_proj.permute(0, 2, 1, 3)
    v_proj = v_proj.permute(0, 2, 1, 3)
    attn_out = F.scaled_dot_product_attention(
        q_proj,
        k_proj,
        v_proj,
        dropout_p=dropout_p,
    )
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch_size, tgt_len, embed_dim)
    attn_out = attn.out_proj(attn_out)
    return attn_out, None


def _build_finite_difference_matrix(
    length: int,
    order: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if order < 1:
        raise ValueError(f"finite-difference order must be >= 1, got {order}")
    if length <= order:
        raise ValueError(f"sequence length must exceed order {order}, got {length}")
    coeffs = torch.tensor(
        [((-1) ** (order - k)) * math.comb(order, k) for k in range(order + 1)],
        dtype=torch.float32,
        device=device,
    )
    rows = length - order
    diff = torch.zeros(rows, length, dtype=torch.float32, device=device)
    for row in range(rows):
        diff[row, row : row + order + 1] = coeffs
    return diff


def _compose_temporal_operator_matrix(
    *,
    length: int,
    operator_type: str,
    velocity_weight: float | torch.Tensor,
    acceleration_weight: float | torch.Tensor,
    jerk_weight: float | torch.Tensor,
) -> torch.Tensor:
    operator_type = str(operator_type).lower()
    operator_device = None
    for weight in (velocity_weight, acceleration_weight, jerk_weight):
        if torch.is_tensor(weight):
            operator_device = weight.device
            break
    operator = torch.zeros(length, length, dtype=torch.float32, device=operator_device)
    velocity_weight_t = (
        velocity_weight.to(device=operator_device, dtype=torch.float32)
        if torch.is_tensor(velocity_weight)
        else torch.tensor(float(velocity_weight), dtype=torch.float32, device=operator_device)
    )
    acceleration_weight_t = (
        acceleration_weight.to(device=operator_device, dtype=torch.float32)
        if torch.is_tensor(acceleration_weight)
        else torch.tensor(float(acceleration_weight), dtype=torch.float32, device=operator_device)
    )
    jerk_weight_t = (
        jerk_weight.to(device=operator_device, dtype=torch.float32)
        if torch.is_tensor(jerk_weight)
        else torch.tensor(float(jerk_weight), dtype=torch.float32, device=operator_device)
    )
    if operator_type in {
        "velocity",
        "velocity_acceleration",
        "velocity_acceleration_jerk",
        "velocity_acceleration_jerk_learned",
    }:
        d1 = _build_finite_difference_matrix(length, order=1, device=operator_device)
        operator = operator + velocity_weight_t * (d1.t() @ d1)
    if operator_type in {
        "acceleration",
        "velocity_acceleration",
        "velocity_acceleration_jerk",
        "velocity_acceleration_jerk_learned",
    }:
        d2 = _build_finite_difference_matrix(length, order=2, device=operator_device)
        operator = operator + acceleration_weight_t * (d2.t() @ d2)
    if operator_type in {
        "jerk",
        "velocity_acceleration_jerk",
        "velocity_acceleration_jerk_learned",
    }:
        d3 = _build_finite_difference_matrix(length, order=3, device=operator_device)
        operator = operator + jerk_weight_t * (d3.t() @ d3)
    return operator


def _build_temporal_operator(
    *,
    length: int,
    operator_type: str,
    q: float,
    velocity_weight: float,
    acceleration_weight: float,
    jerk_weight: float,
    jitter: float,
    spectral_transform: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    operator_type = str(operator_type).lower()
    spectral_transform = str(spectral_transform).lower()
    if operator_type == "identity":
        basis = torch.eye(length, dtype=torch.float32)
        return basis, basis, torch.ones(length, dtype=torch.float32)

    if operator_type == "dct_frequency":
        temporal_dct = _build_dct_matrix(length).to(torch.float32)
        temporal_basis = temporal_dct.t()
        omega = torch.arange(length, dtype=torch.float32)
        temporal_eigvals = torch.pow(1.0 + omega.pow(2), float(q))
        return temporal_basis, temporal_dct, temporal_eigvals

    operator = _compose_temporal_operator_matrix(
        length=length,
        operator_type=operator_type,
        velocity_weight=velocity_weight,
        acceleration_weight=acceleration_weight,
        jerk_weight=jerk_weight,
    )

    if not torch.any(operator):
        raise ValueError(
            "temporal operator must contribute non-zero energy. "
            f"Got operator_type={operator_type!r}, velocity_weight={velocity_weight}, "
            f"acceleration_weight={acceleration_weight}, jerk_weight={jerk_weight}."
        )

    base_operator = operator
    if spectral_transform == "identity_shifted_power":
        operator = operator + max(float(jitter), 1e-8) * torch.eye(
            length,
            dtype=torch.float32,
            device=operator.device,
        )
    elif spectral_transform != "matrix_exp":
        raise ValueError(
            "temporal spectral transform must be one of "
            "{'identity_shifted_power', 'matrix_exp'}, "
            f"got {spectral_transform!r}"
        )

    eigvals, eigvecs = torch.linalg.eigh(operator)
    eigvals = torch.clamp(eigvals, min=0.0)
    if spectral_transform == "identity_shifted_power":
        temporal_eigvals = torch.pow(1.0 + eigvals, float(q))
    elif spectral_transform == "matrix_exp":
        base_eigvals = torch.linalg.eigvalsh(base_operator)
        base_eigvals = torch.clamp(base_eigvals, min=0.0)
        temporal_eigvals = torch.pow(torch.exp(float(jitter) * base_eigvals), float(q))
    return eigvecs.to(torch.float32), eigvecs.t().to(torch.float32), temporal_eigvals.to(torch.float32)


def _build_frequency_operator_eigvals(
    *,
    length: int,
    operator_type: str,
    q: float,
    velocity_weight: float | torch.Tensor,
    acceleration_weight: float | torch.Tensor,
    jerk_weight: float | torch.Tensor,
) -> torch.Tensor:
    """Build frequency-bin diffusion rates for a DCT-domain state."""
    operator_type = str(operator_type).lower()
    operator_device = None
    for weight in (velocity_weight, acceleration_weight, jerk_weight):
        if torch.is_tensor(weight):
            operator_device = weight.device
            break
    omega = torch.arange(length, dtype=torch.float32, device=operator_device)
    velocity_weight_t = (
        velocity_weight.to(device=operator_device, dtype=torch.float32)
        if torch.is_tensor(velocity_weight)
        else torch.tensor(float(velocity_weight), dtype=torch.float32, device=operator_device)
    )
    acceleration_weight_t = (
        acceleration_weight.to(device=operator_device, dtype=torch.float32)
        if torch.is_tensor(acceleration_weight)
        else torch.tensor(float(acceleration_weight), dtype=torch.float32, device=operator_device)
    )
    jerk_weight_t = (
        jerk_weight.to(device=operator_device, dtype=torch.float32)
        if torch.is_tensor(jerk_weight)
        else torch.tensor(float(jerk_weight), dtype=torch.float32, device=operator_device)
    )

    if operator_type == "identity":
        return torch.ones(length, dtype=torch.float32, device=operator_device)
    if operator_type == "dct_frequency":
        energy = 1.0 + omega.pow(2)
    elif operator_type == "velocity":
        energy = 1.0 + velocity_weight_t * omega.pow(2)
    elif operator_type == "acceleration":
        energy = 1.0 + acceleration_weight_t * omega.pow(4)
    elif operator_type == "jerk":
        energy = 1.0 + jerk_weight_t * omega.pow(6)
    elif operator_type in {"velocity_acceleration", "velocity_acceleration_jerk", "velocity_acceleration_jerk_learned"}:
        energy = 1.0 + velocity_weight_t * omega.pow(2) + acceleration_weight_t * omega.pow(4)
        if operator_type in {"velocity_acceleration_jerk", "velocity_acceleration_jerk_learned"}:
            energy = energy + jerk_weight_t * omega.pow(6)
    else:
        raise ValueError(
            "frequency operator must be one of "
            "{'identity', 'dct_frequency', 'velocity', 'acceleration', 'jerk', "
            "'velocity_acceleration', 'velocity_acceleration_jerk', "
            "'velocity_acceleration_jerk_learned'}, "
            f"got {operator_type!r}"
        )
    return torch.pow(torch.clamp(energy, min=1e-8), float(q)).to(torch.float32)


# ----------------------------- Configs ----------------------------------------

@dataclass
class CardConfig:
    input_length: int
    pred_length: int
    num_nodes: int

    # Coarse backbone (siMLPe-like)
    hidden_dim: int
    num_layers: int
    simlpe_use_norm: bool = True
    simlpe_spatial_fc_only: bool = False
    simlpe_mix_spatial_temporal: bool = False
    simlpe_norm_axis: str = "spatial"
    simlpe_add_last_offset: bool = True

    # DCT split (still used by coarse)
    k_low: int = 16

    # Diffusion
    diffusion_steps: int = 100
    ddim_steps: int = 50
    beta_schedule: str = "cosine"  # "cosine" or "linear"
    isotropic_noise: bool = False
    spatial_anisotropy: bool = True
    beta_matrix_power: float = 1.0
    beta_matrix_min_rate: Optional[float] = 0.5
    beta_matrix_max_rate: Optional[float] = 2.0
    temporal_anisotropy: bool = False
    temporal_anisotropy_q: float = 1.0
    temporal_operator_type: str = "identity"
    temporal_operator_spectral_transform: str = "identity_shifted_power"
    temporal_velocity_weight: float = 1.0
    temporal_acceleration_weight: float = 1.0
    temporal_jerk_weight: float = 1.0
    temporal_anisotropy_learned_from_history: bool = False
    temporal_anisotropy_history_dim: int = 128
    temporal_anisotropy_delta_max_abs: float = 2.0

    # Denoiser architecture
    denoiser_dim: int = 256
    denoiser_depth: int = 6
    denoiser_heads: int = 8
    dropout: float = 0.0

    # Training
    freeze_coarse: bool = True
    stopgrad_coarse_condition: bool = True
    cond_use_history: bool = True
    cond_use_coarse: bool = True
    allow_no_conditioning: bool = False
    coarse_target_lowpass_only: bool = False
    diffusion_only: bool = False

    # Sampling stabilizers
    x0_clip: float = 0  # clip x0_hat in normalized space (0 disables)

    # Structured covariance knobs
    node_covariance_type: str = "laplacian_heat_kernel"
    mobility_palm_var: float = 0.15
    mobility_depth1_var: float = 0.35
    mobility_depth2_var: float = 0.70
    mobility_depth3plus_var: float = 1.00
    dhalf_gamma: float = 1.0
    learnable_dhalf: bool = False
    graph_laplacian_tau: float = 1.0
    graph_laplacian_alpha: float = 0.0
    graph_laplacian_beta: float = 1.0
    graph_laplacian_normalized: bool = True
    covariance_jitter: float = 1e-4

    @property
    def total_length(self) -> int:
        return int(self.input_length + self.pred_length)

    @property
    def feature_dim(self) -> int:
        return int(self.num_nodes * 3)

    @property
    def k_high(self) -> int:
        return int(max(0, self.total_length - self.k_low))


# ----------------------------- Stage A ----------------------------------------

class CoarseDCTForecaster(nn.Module):
    """Predict low-frequency DCT coefficients and reconstruct a coarse future."""

    def __init__(self, cfg: CardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        T = cfg.total_length
        if not (1 <= cfg.k_low <= T):
            raise ValueError(f"k_low must be in [1, {T}], got {cfg.k_low}")

        dct = _build_dct_matrix(T)
        idct = dct.t()
        future_dct = _build_dct_matrix(cfg.pred_length)
        self.register_buffer("_dct_low", dct[: cfg.k_low, :], persistent=False)
        self.register_buffer("_idct_low", idct[:, : cfg.k_low], persistent=False)
        self.register_buffer("_future_dct", future_dct, persistent=False)

        sim_cfg = SiMLPeConfig(
            input_length=cfg.input_length,
            pred_length=cfg.pred_length,
            num_nodes=cfg.num_nodes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_norm=cfg.simlpe_use_norm,
            use_spatial_fc_only=cfg.simlpe_spatial_fc_only,
            mix_spatial_temporal=cfg.simlpe_mix_spatial_temporal,
            norm_axis=cfg.simlpe_norm_axis,
            add_last_offset=cfg.simlpe_add_last_offset,
        )
        self.backbone = SiMLPeBackbone(sim_cfg)

    def _pad_total(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        B = history.size(0)
        flat = history.reshape(B, self.cfg.input_length, -1)
        last = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
        padded = torch.cat([flat, last], dim=1)
        return padded

    def predict_low_coeffs(self, history: torch.Tensor) -> torch.Tensor:
        padded = self._pad_total(history)
        dct_low = self._dct_low.unsqueeze(0).to(device=padded.device, dtype=padded.dtype)
        low_coeffs = torch.matmul(dct_low, padded)

        B, _, F = low_coeffs.shape
        coeffs_full = torch.zeros(B, self.cfg.total_length, F, device=padded.device, dtype=padded.dtype)
        coeffs_full[:, : self.cfg.k_low, :] = low_coeffs

        pred_full = self.backbone(coeffs_full)
        pred_low = pred_full[:, : self.cfg.k_low, :]
        return pred_low

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        future_flat = self.predict_future_flat(history)
        future = future_flat.reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )
        return future

    def predict_future_flat(self, history: torch.Tensor) -> torch.Tensor:
        pred_low = self.predict_low_coeffs(history)
        idct_low_future = self._idct_low[-self.cfg.pred_length :, :].unsqueeze(0).to(
            device=pred_low.device,
            dtype=pred_low.dtype,
        )
        future = torch.matmul(idct_low_future, pred_low)
        if self.cfg.simlpe_add_last_offset:
            last = history.reshape(history.size(0), self.cfg.input_length, -1)[:, -1:, :]
            future = future + last
        return future

    def predict_future_coeffs(self, history: torch.Tensor) -> torch.Tensor:
        future = self.predict_future_flat(history)
        future_dct = self._future_dct.unsqueeze(0).to(device=future.device, dtype=future.dtype)
        return torch.matmul(future_dct, future)

    def lowpass_future_target(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        _assert_shape(future, (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3), "future")
        full = torch.cat([history, future], dim=1).reshape(history.size(0), self.cfg.total_length, -1)
        dct_low = self._dct_low.unsqueeze(0).to(device=full.device, dtype=full.dtype)
        idct_low = self._idct_low.unsqueeze(0).to(device=full.device, dtype=full.dtype)
        seq_low = torch.matmul(idct_low, torch.matmul(dct_low, full))
        future_low = seq_low[:, -self.cfg.pred_length :, :].reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )
        if self.cfg.simlpe_add_last_offset:
            future_low = future_low + history[:, -1:, :, :]
        return future_low


# ----------------------------- Diffusion scheduler ----------------------------

class DiffusionSchedule:
    def __init__(self, steps: int, schedule: str = "cosine", device: Optional[torch.device] = None) -> None:
        if steps <= 1:
            raise ValueError("diffusion steps must be > 1")
        self.steps = int(steps)
        self.schedule = str(schedule)
        self.device = device

        betas = self._make_betas(self.steps, self.schedule)
        self.betas = betas.to(device=device)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @staticmethod
    def _make_betas(T: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        if schedule == "cosine":
            s = 0.008
            steps = T
            x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 1e-8, 0.999).float()
            return betas
        raise ValueError(f"Unknown beta schedule: {schedule}")


def _index_schedule_rows(table: torch.Tensor, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1)
    if table.ndim == 4:
        table = table.to(device=timesteps.device, dtype=dtype)
        if int(table.shape[1]) != int(timesteps.numel()):
            raise ValueError(
                "Batched spectral schedule expects table.shape[1] to match the batch size, "
                f"got {tuple(table.shape)} and batch {timesteps.numel()}"
            )
        batch_idx = torch.arange(timesteps.numel(), device=timesteps.device)
        return table[timesteps, batch_idx]
    return torch.index_select(table.to(device=timesteps.device), 0, timesteps).to(dtype=dtype)


# ----------------------------- Denoiser (factorized ST with joint tokens) -----

class _FiLM(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shape = [cond.size(0)] + [1] * (x.ndim - 2) + [cond.size(-1)]
        scale = self.to_scale(cond).view(*shape)
        shift = self.to_shift(cond).view(*shape)
        return x * (1.0 + scale) + shift


class _MLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        y = self.norm1(x).reshape(B * T, N, D)
        y, _ = _multihead_attention_fast(self.attn, y, y, y)
        x = x + y.reshape(B, T, N, D)
        x = x + self.mlp(self.norm2(x))
        return x


class _TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        y = self.norm1(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        y, _ = _multihead_attention_fast(self.attn, y, y, y)
        y = y.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class _TemporalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, T_out, N, D = x.shape
        _, T_mem, N_mem, D_mem = memory.shape
        if N_mem != N or D_mem != D:
            raise ValueError(
                f"memory shape mismatch: expected joint dim {N} and feature dim {D}, "
                f"got {(B, T_mem, N_mem, D_mem)}"
            )
        if T_mem == 0:
            x = x + self.mlp(self.norm2(x))
            return x
        q = self.norm_q(x).permute(0, 2, 1, 3).reshape(B * N, T_out, D)
        m = self.norm_m(memory).permute(0, 2, 1, 3).reshape(B * N, T_mem, D)
        y, _ = _multihead_attention_fast(self.attn, q, m, m)
        y = y.reshape(B, N, T_out, D).permute(0, 2, 1, 3)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class _FactorizedSTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.film_spatial = _FiLM(dim, dim)
        self.spatial = _SpatialSelfAttentionBlock(dim, n_heads, dropout)

        self.film_temporal = _FiLM(dim, dim)
        self.temporal = _TemporalSelfAttentionBlock(dim, n_heads, dropout)

        self.film_cross = _FiLM(dim, dim)
        self.cross = _TemporalCrossAttentionBlock(dim, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.film_spatial(x, t_emb)
        x = self.spatial(x)
        x = self.film_temporal(x, t_emb)
        x = self.temporal(x)
        x = self.film_cross(x, t_emb)
        x = self.cross(x, memory)
        return x


class MRTimeTransformerDenoiser(nn.Module):
    """Factorized spatio-temporal denoiser with explicit joint tokens."""

    def __init__(
        self,
        *,
        in_feat: int,
        t_in: int,
        t_out: int,
        d_model: int,
        depth: int,
        n_heads: int,
        dropout: float,
        cond_use_history: bool = True,
        cond_use_coarse: bool = True,
        allow_no_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.in_feat = int(in_feat)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.d_model = int(d_model)
        self.cond_use_history = bool(cond_use_history)
        self.cond_use_coarse = bool(cond_use_coarse)
        self.allow_no_conditioning = bool(allow_no_conditioning)

        if self.in_feat % 3 != 0:
            raise ValueError(f"in_feat must be divisible by 3, got {self.in_feat}")
        self.num_nodes = self.in_feat // 3

        self.proj_q = nn.Linear(3, d_model)
        self.proj_hist = nn.Linear(3, d_model)
        self.proj_coarse = nn.Linear(3, d_model)
        self.proj_mamp = nn.LazyLinear(d_model)

        self.pos_q_time = nn.Parameter(torch.zeros(1, t_out, 1, d_model))
        self.pos_hist_time = nn.Parameter(torch.zeros(1, t_in, 1, d_model))
        self.pos_coarse_time = nn.Parameter(torch.zeros(1, t_out, 1, d_model))
        self.pos_joint = nn.Parameter(torch.zeros(1, 1, self.num_nodes, d_model))
        self.pos_mamp = nn.Parameter(torch.zeros(1, 1, d_model))

        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mamp_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, depth // 2))

        self.blocks = nn.ModuleList([
            _FactorizedSTBlock(d_model, n_heads, dropout) for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 3)

        nn.init.normal_(self.pos_q_time, std=0.02)
        nn.init.normal_(self.pos_hist_time, std=0.02)
        nn.init.normal_(self.pos_coarse_time, std=0.02)
        nn.init.normal_(self.pos_joint, std=0.02)
        nn.init.normal_(self.pos_mamp, std=0.02)

    def _flat_to_joint_tokens(self, x: torch.Tensor, T: int, name: str) -> torch.Tensor:
        B = x.size(0)
        _assert_shape(x, (B, T, self.in_feat), name)
        return x.reshape(B, T, self.num_nodes, 3)

    def _encode_mamp_per_joint(
        self,
        mamp_feat: Optional[torch.Tensor],
        *,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mamp_feat is None:
            return None
        if int(mamp_feat.shape[0]) != B:
            raise ValueError(f"mamp_feat batch mismatch: expected {B}, got {mamp_feat.shape[0]}")
        if mamp_feat.ndim == 2:
            token = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype)).unsqueeze(1) + self.pos_mamp
            token = token.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
            return token
        if mamp_feat.ndim == 3:
            if int(mamp_feat.shape[1]) == self.num_nodes:
                joint_tok = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype))
                return joint_tok.unsqueeze(1)
            tok = self.proj_mamp(mamp_feat.to(device=device, dtype=dtype))
            tok = tok + self.pos_mamp
            tok = self.mamp_encoder(tok)
            tok = tok.mean(dim=1, keepdim=True)
            tok = tok.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
            return tok
        raise ValueError(
            f"mamp_feat must be (B, D), (B, K, D), or (B, N, D), got {tuple(mamp_feat.shape)}"
        )

    def forward(
        self,
        x_noisy: torch.Tensor,        # (B, T_out, F)
        timesteps: torch.Tensor,      # (B,)
        history: torch.Tensor,        # (B, T_in, F)
        coarse_future: torch.Tensor,  # (B, T_out, F)
        mamp_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_noisy.size(0)
        _assert_shape(x_noisy, (B, self.t_out, self.in_feat), "x_noisy")
        _assert_shape(history, (B, self.t_in, self.in_feat), "history_flat")
        _assert_shape(coarse_future, (B, self.t_out, self.in_feat), "coarse_future_flat")

        x_noisy_j = self._flat_to_joint_tokens(x_noisy, self.t_out, "x_noisy")
        history_j = self._flat_to_joint_tokens(history, self.t_in, "history_flat")
        coarse_j = self._flat_to_joint_tokens(coarse_future, self.t_out, "coarse_future_flat")

        x = self.proj_q(x_noisy_j) + self.pos_q_time + self.pos_joint

        memory_parts = []
        if self.cond_use_history:
            h_hist = self.proj_hist(history_j) + self.pos_hist_time + self.pos_joint
            memory_parts.append(h_hist)
        if self.cond_use_coarse:
            h_coarse = self.proj_coarse(coarse_j) + self.pos_coarse_time + self.pos_joint
            memory_parts.append(h_coarse)

        if not memory_parts and mamp_feat is None and not self.allow_no_conditioning:
            raise ValueError("No conditioning enabled.")

        if memory_parts:
            memory = torch.cat(memory_parts, dim=1)
        else:
            memory = torch.zeros(B, 0, self.num_nodes, self.d_model, device=x.device, dtype=x.dtype)

        mamp_joint = self._encode_mamp_per_joint(mamp_feat, B=B, device=x.device, dtype=x.dtype)
        if mamp_joint is not None:
            x = x + mamp_joint
            if memory.size(1) > 0:
                memory = memory + mamp_joint.expand(-1, memory.size(1), -1, -1)

        te = _timestep_embedding(timesteps, self.d_model).to(dtype=x.dtype, device=x.device)
        te = self.t_embed(te)

        for block in self.blocks:
            x = block(x, memory, te)

        v_hat = self.out(self.out_norm(x))
        v_hat = v_hat.reshape(B, self.t_out, self.in_feat)
        return v_hat


# ----------------------------- Hand covariance builder ------------------------

class HandKinematicCovariance:
    def __init__(
        self,
        *,
        num_nodes: int,
        wrist_index: int,
        edges: Iterable[Tuple[int, int]],
        node_covariance_type: str,
        palm_var: float,
        depth1_var: float,
        depth2_var: float,
        depth3plus_var: float,
        dhalf_gamma: float,
        laplacian_tau: float,
        laplacian_alpha: float,
        laplacian_beta: float,
        laplacian_normalized: bool,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.wrist_index = int(wrist_index)
        self.edges = [(int(i), int(j)) for i, j in edges]
        self.node_covariance_type = str(node_covariance_type)
        self.palm_var = float(palm_var)
        self.depth1_var = float(depth1_var)
        self.depth2_var = float(depth2_var)
        self.depth3plus_var = float(depth3plus_var)
        self.dhalf_gamma = float(dhalf_gamma)
        self.laplacian_tau = float(laplacian_tau)
        self.laplacian_alpha = float(laplacian_alpha)
        self.laplacian_beta = float(laplacian_beta)
        self.laplacian_normalized = bool(laplacian_normalized)

        if not (0 <= self.wrist_index < self.num_nodes):
            raise ValueError(f"Invalid wrist_index={self.wrist_index} for num_nodes={self.num_nodes}")
        if self.laplacian_tau < 0.0:
            raise ValueError(f"laplacian_tau must be non-negative, got {self.laplacian_tau}")
        if self.laplacian_alpha < 0.0:
            raise ValueError(f"laplacian_alpha must be non-negative, got {self.laplacian_alpha}")
        if self.laplacian_beta < 0.0:
            raise ValueError(f"laplacian_beta must be non-negative, got {self.laplacian_beta}")
        if self.dhalf_gamma < 0.0:
            raise ValueError(f"dhalf_gamma must be non-negative, got {self.dhalf_gamma}")
        if self.node_covariance_type not in {
            "laplacian_heat_kernel",
            "gaussian_kernel",
            "empirical_temporal_mean",
            "empirical_heat_kernel",
            "empirical_plus_theoretical",
            "laplacian_only",
            "diagonal",
        }:
            raise ValueError(
                "node_covariance_type must be one of "
                f"{{'laplacian_heat_kernel', 'gaussian_kernel', 'empirical_temporal_mean', "
                f"'empirical_heat_kernel', "
                f"'empirical_plus_theoretical', "
                f"'laplacian_only', 'diagonal'}}, "
                f"got {self.node_covariance_type!r}"
            )

        self.free_indices = [i for i in range(self.num_nodes) if i != self.wrist_index]
        self.num_free_nodes = len(self.free_indices)

    def _make_adjacency_full(self) -> List[List[int]]:
        adj = [[] for _ in range(self.num_nodes)]
        for i, j in self.edges:
            if not (0 <= i < self.num_nodes and 0 <= j < self.num_nodes):
                raise ValueError(f"Edge {(i, j)} out of bounds for num_nodes={self.num_nodes}")
            if j not in adj[i]:
                adj[i].append(j)
            if i not in adj[j]:
                adj[j].append(i)
        return adj

    def _adjacency_matrix_full(self, device: Optional[torch.device] = None) -> torch.Tensor:
        A1 = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32, device=device)
        full_adj = self._make_adjacency_full()
        for i_orig in range(self.num_nodes):
            for j_orig in full_adj[i_orig]:
                A1[i_orig, j_orig] = 1.0
        A1 = torch.maximum(A1, A1.t())
        A1.fill_diagonal_(0.0)
        return A1

    def _dist_from_wrist(self) -> List[int]:
        adj = self._make_adjacency_full()
        dist = [-1] * self.num_nodes
        q: deque[int] = deque([self.wrist_index])
        dist[self.wrist_index] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if any(d < 0 for d in dist):
            raise ValueError("Hand graph is disconnected from the wrist.")
        return dist

    def _node_variances_free(
        self,
        bucket_variances: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        dist = self._dist_from_wrist()
        if bucket_variances is None:
            bucket_values = torch.tensor(
                [
                    self.palm_var,
                    self.depth1_var,
                    self.depth2_var,
                    self.depth3plus_var,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            if bucket_variances.shape != (4,):
                raise ValueError(f"bucket_variances must have shape (4,), got {tuple(bucket_variances.shape)}")
            bucket_values = bucket_variances.to(device=device, dtype=torch.float32)
        bucket_indices: List[int] = []
        for orig_idx in self.free_indices:
            d = dist[orig_idx]
            if d <= 1:
                bucket_indices.append(0)
            elif d == 2:
                bucket_indices.append(1)
            elif d == 3:
                bucket_indices.append(2)
            else:
                bucket_indices.append(3)
        index_tensor = torch.tensor(bucket_indices, dtype=torch.long, device=bucket_values.device)
        return bucket_values.index_select(0, index_tensor)

    def _graph_laplacian_full(self, device: Optional[torch.device] = None) -> torch.Tensor:
        A1 = self._adjacency_matrix_full(device=device)
        deg = A1.sum(dim=1)
        D = torch.diag(deg)
        eye = torch.eye(self.num_nodes, dtype=torch.float32, device=A1.device)
        if self.laplacian_normalized:
            inv_sqrt_deg = torch.zeros_like(deg)
            mask = deg > 0
            inv_sqrt_deg[mask] = deg[mask].rsqrt()
            D_inv_sqrt = torch.diag(inv_sqrt_deg)
            laplacian = eye - (D_inv_sqrt @ A1 @ D_inv_sqrt)
        else:
            laplacian = D - A1
        return self.laplacian_alpha * eye + self.laplacian_beta * laplacian

    def _laplacian_heat_kernel_full(self, device: Optional[torch.device] = None) -> torch.Tensor:
        lap = self._graph_laplacian_full(device=device)
        eigvals, eigvecs = torch.linalg.eigh(lap)
        heat_eigvals = torch.exp(-self.laplacian_tau * eigvals)
        return eigvecs @ torch.diag(heat_eigvals) @ eigvecs.t()

    def _gaussian_kernel_full(self, device: Optional[torch.device] = None) -> torch.Tensor:
        lap = self._graph_laplacian_full(device=device)
        eigvals, eigvecs = torch.linalg.eigh(lap)
        gaussian_eigvals = torch.exp(-0.5 * (self.laplacian_tau ** 2) * (eigvals ** 2))
        return eigvecs @ torch.diag(gaussian_eigvals) @ eigvecs.t()

    def _feature_scale_half(
        self,
        node_variances: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if node_variances is None:
            node_variances = self._node_variances_free(device=device)
        feature_variances = node_variances.repeat_interleave(3)
        return torch.diag(torch.pow(feature_variances, 0.5 * self.dhalf_gamma))

    def _empirical_feature_covariance_scaled(
        self,
        empirical_feature_covariance: torch.Tensor,
        node_variances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat_dim = 3 * self.num_free_nodes
        if empirical_feature_covariance.shape != (feat_dim, feat_dim):
            raise ValueError(
                "empirical_feature_covariance must have shape "
                f"({feat_dim}, {feat_dim}), got {tuple(empirical_feature_covariance.shape)}"
            )
        D_half = self._feature_scale_half(
            node_variances=node_variances,
            device=empirical_feature_covariance.device,
        ).to(empirical_feature_covariance.dtype)
        return D_half @ empirical_feature_covariance @ D_half

    def _empirical_feature_covariance_unscaled(self, empirical_feature_covariance: torch.Tensor) -> torch.Tensor:
        feat_dim = 3 * self.num_free_nodes
        if empirical_feature_covariance.shape != (feat_dim, feat_dim):
            raise ValueError(
                "empirical_feature_covariance must have shape "
                f"({feat_dim}, {feat_dim}), got {tuple(empirical_feature_covariance.shape)}"
            )
        return empirical_feature_covariance

    def _empirical_heat_kernel_covariance(
        self,
        empirical_feature_covariance: torch.Tensor,
        node_variances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        empirical_covariance = self._empirical_feature_covariance_unscaled(empirical_feature_covariance)
        eigvals, eigvecs = torch.linalg.eigh(empirical_covariance)
        heat_eigvals = torch.exp(-self.laplacian_tau * eigvals)
        heat_kernel = eigvecs @ torch.diag(heat_eigvals) @ eigvecs.t()
        D_half = self._feature_scale_half(
            node_variances=node_variances,
            device=heat_kernel.device,
        ).to(heat_kernel.dtype)
        return D_half @ heat_kernel @ D_half

    def build_feature_covariance(
        self,
        empirical_feature_covariance: Optional[torch.Tensor] = None,
        bucket_variances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Nf = self.num_free_nodes
        if Nf <= 0:
            raise ValueError("No free joints available after removing wrist.")
        target_device = None
        if bucket_variances is not None:
            target_device = bucket_variances.device
        elif empirical_feature_covariance is not None:
            target_device = empirical_feature_covariance.device
        node_variances = self._node_variances_free(bucket_variances=bucket_variances, device=target_device)
        if self.node_covariance_type == "empirical_temporal_mean":
            if empirical_feature_covariance is None:
                raise ValueError("empirical_feature_covariance is required for empirical_temporal_mean covariance.")
            return self._empirical_feature_covariance_scaled(
                empirical_feature_covariance,
                node_variances=node_variances,
            )
        if self.node_covariance_type == "empirical_heat_kernel":
            if empirical_feature_covariance is None:
                raise ValueError("empirical_feature_covariance is required for empirical_heat_kernel covariance.")
            return self._empirical_heat_kernel_covariance(
                empirical_feature_covariance,
                node_variances=node_variances,
            )

        if self.node_covariance_type == "diagonal":
            Sigma_node = torch.diag(torch.pow(node_variances, self.dhalf_gamma))
        else:
            if self.node_covariance_type == "gaussian_kernel":
                Sigma_node_full = self._gaussian_kernel_full(device=node_variances.device)
            else:
                Sigma_node_full = self._laplacian_heat_kernel_full(device=node_variances.device)
            Sigma_node = Sigma_node_full[self.free_indices][:, self.free_indices]
            if self.node_covariance_type in {"laplacian_heat_kernel", "gaussian_kernel"}:
                D_half = torch.diag(torch.pow(node_variances, 0.5 * self.dhalf_gamma))
                Sigma_node = D_half @ Sigma_node @ D_half
        theoretical_covariance = torch.kron(
            Sigma_node,
            torch.eye(3, dtype=torch.float32, device=Sigma_node.device),
        )
        if self.node_covariance_type == "empirical_plus_theoretical":
            if empirical_feature_covariance is None:
                raise ValueError(
                    "empirical_feature_covariance is required for empirical_plus_theoretical covariance."
                )
            empirical_covariance = self._empirical_feature_covariance_scaled(
                empirical_feature_covariance,
                node_variances=node_variances,
            )
            return theoretical_covariance + empirical_covariance
        return theoretical_covariance


# ----------------------------- Diffusion module -------------------------------

class TimeResidualConditionalDiffusion(nn.Module):
    def __init__(self, cfg: CardConfig, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.schedule = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_schedule)

        metadata = dict(metadata or {})
        if "wrist_index" not in metadata:
            raise ValueError("Missing required metadata['wrist_index'].")
        self.wrist_index = int(metadata["wrist_index"])
        edges_raw = metadata.get("edges", ())
        self.edges = [(int(i), int(j)) for i, j in edges_raw] if edges_raw else []
        empirical_feature_covariance = metadata.get("empirical_feature_covariance")
        if empirical_feature_covariance is not None:
            empirical_feature_covariance = torch.as_tensor(empirical_feature_covariance, dtype=torch.float32)

        if not (0 <= self.wrist_index < cfg.num_nodes):
            raise ValueError(f"Invalid wrist_index={self.wrist_index} for num_nodes={cfg.num_nodes}")
        self.free_indices = [i for i in range(cfg.num_nodes) if i != self.wrist_index]
        self.num_free_nodes = len(self.free_indices)
        self.free_feature_dim = 3 * self.num_free_nodes
        self.learnable_dhalf = bool(cfg.learnable_dhalf)
        self.cov_builder: Optional[HandKinematicCovariance] = None
        self.temporal_anisotropy = bool(cfg.temporal_anisotropy)
        self.temporal_anisotropy_learned_from_history = bool(cfg.temporal_anisotropy_learned_from_history)
        future_dct = _build_dct_matrix(cfg.pred_length)
        self.register_buffer("future_dct", future_dct.to(torch.float32), persistent=True)
        self.register_buffer("future_idct", future_dct.t().to(torch.float32), persistent=True)

        if cfg.isotropic_noise or not cfg.spatial_anisotropy or self.num_free_nodes <= 0:
            cov = torch.eye(max(self.free_feature_dim, 1), dtype=torch.float32)
            if self.free_feature_dim == 0:
                cov = cov[:0, :0]
        elif self.edges:
            cov_builder = HandKinematicCovariance(
                num_nodes=cfg.num_nodes,
                wrist_index=self.wrist_index,
                edges=self.edges,
                node_covariance_type=cfg.node_covariance_type,
                palm_var=cfg.mobility_palm_var,
                depth1_var=cfg.mobility_depth1_var,
                depth2_var=cfg.mobility_depth2_var,
                depth3plus_var=cfg.mobility_depth3plus_var,
                dhalf_gamma=cfg.dhalf_gamma,
                laplacian_tau=cfg.graph_laplacian_tau,
                laplacian_alpha=cfg.graph_laplacian_alpha,
                laplacian_beta=cfg.graph_laplacian_beta,
                laplacian_normalized=cfg.graph_laplacian_normalized,
            )
            self.cov_builder = cov_builder
            cov = cov_builder.build_feature_covariance(empirical_feature_covariance=empirical_feature_covariance)
        else:
            cov = torch.eye(self.free_feature_dim, dtype=torch.float32)
        self._empirical_feature_covariance = empirical_feature_covariance

        temporal_operator_type = str(cfg.temporal_operator_type).lower()
        if temporal_operator_type == "identity" and self.temporal_anisotropy:
            temporal_operator_type = "dct_frequency"
        self.temporal_operator_type = temporal_operator_type
        self.learned_temporal_operator_coefficients = (
            temporal_operator_type == "velocity_acceleration_jerk_learned"
        )
        temporal_basis = torch.eye(cfg.pred_length, dtype=torch.float32)
        temporal_basis_t = temporal_basis
        if self.temporal_anisotropy:
            temporal_eigvals = _build_frequency_operator_eigvals(
                length=cfg.pred_length,
                operator_type=temporal_operator_type,
                q=cfg.temporal_anisotropy_q,
                velocity_weight=cfg.temporal_velocity_weight,
                acceleration_weight=cfg.temporal_acceleration_weight,
                jerk_weight=cfg.temporal_jerk_weight,
            )
        else:
            temporal_eigvals = torch.ones(cfg.pred_length, dtype=torch.float32)
        if self.learned_temporal_operator_coefficients:
            temporal_weight_init = torch.tensor(
                [
                    max(float(cfg.temporal_velocity_weight), 1e-6),
                    max(float(cfg.temporal_acceleration_weight), 1e-6),
                    max(float(cfg.temporal_jerk_weight), 1e-6),
                ],
                dtype=torch.float32,
            )
            temporal_weight_init = torch.log(torch.expm1(temporal_weight_init))
            self.temporal_operator_weight_raw = nn.Parameter(temporal_weight_init)
        else:
            self.register_parameter("temporal_operator_weight_raw", None)

        if self.learnable_dhalf:
            bucket_init = torch.tensor(
                [
                    cfg.mobility_palm_var,
                    cfg.mobility_depth1_var,
                    cfg.mobility_depth2_var,
                    cfg.mobility_depth3plus_var,
                ],
                dtype=torch.float32,
            )
            raw_init = torch.log(torch.expm1(torch.clamp(bucket_init, min=1e-6)))
            self.dhalf_bucket_raw = nn.Parameter(raw_init)
        else:
            self.register_parameter("dhalf_bucket_raw", None)

        if self.temporal_anisotropy and self.temporal_anisotropy_learned_from_history:
            hist_dim = max(8, int(cfg.temporal_anisotropy_history_dim))
            self.temporal_history_encoder = nn.Sequential(
                nn.LayerNorm(self.free_feature_dim),
                nn.Linear(self.free_feature_dim, hist_dim),
                nn.GELU(),
            )
            self.temporal_delta_mlp = nn.Sequential(
                nn.LayerNorm(hist_dim),
                nn.Linear(hist_dim, hist_dim),
                nn.GELU(),
                nn.Linear(hist_dim, cfg.pred_length),
            )
            nn.init.zeros_(self.temporal_delta_mlp[-1].weight)
            nn.init.zeros_(self.temporal_delta_mlp[-1].bias)
        else:
            self.temporal_history_encoder = None
            self.temporal_delta_mlp = None

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=max(cfg.covariance_jitter, 1e-8))
        mode_rates, spectral_alpha_bars = self._build_spatiotemporal_schedule(
            node_eigvals=eigvals,
            temporal_eigvals=temporal_eigvals,
            device=eigvals.device,
        )
        spectral_alpha_bars = torch.clamp(spectral_alpha_bars, min=1e-8, max=1.0)
        spectral_alpha_bars_prev = torch.cat(
            [torch.ones(1, cfg.pred_length, self.free_feature_dim, dtype=torch.float32), spectral_alpha_bars[:-1]],
            dim=0,
        )
        spectral_alphas = torch.clamp(spectral_alpha_bars / spectral_alpha_bars_prev, min=1e-8, max=0.999999)
        self.register_buffer("temporal_basis", temporal_basis.to(torch.float32), persistent=True)
        self.register_buffer("temporal_basis_t", temporal_basis_t.to(torch.float32), persistent=True)
        self.register_buffer("temporal_mode_eigvals", temporal_eigvals.to(torch.float32), persistent=True)
        self.register_buffer("cov_base", cov.to(torch.float32), persistent=True)
        self.register_buffer("beta_basis", eigvecs.to(torch.float32), persistent=True)
        self.register_buffer("beta_basis_t", eigvecs.t().to(torch.float32), persistent=True)
        self.register_buffer("beta_mode_eigvals", eigvals.to(torch.float32), persistent=True)
        self.register_buffer("beta_mode_rates", mode_rates.to(torch.float32), persistent=True)
        self.register_buffer("spectral_alpha_bars", spectral_alpha_bars.to(torch.float32), persistent=True)
        self.register_buffer("spectral_alphas", spectral_alphas.to(torch.float32), persistent=True)
        self.denoiser = MRTimeTransformerDenoiser(
            in_feat=self.free_feature_dim,
            t_in=cfg.input_length,
            t_out=cfg.pred_length,
            d_model=cfg.denoiser_dim,
            depth=cfg.denoiser_depth,
            n_heads=cfg.denoiser_heads,
            dropout=cfg.dropout,
            cond_use_history=cfg.cond_use_history,
            cond_use_coarse=cfg.cond_use_coarse,
            allow_no_conditioning=cfg.allow_no_conditioning,
        )

    def _current_bucket_variances(self) -> Optional[torch.Tensor]:
        if not self.learnable_dhalf or self.dhalf_bucket_raw is None:
            return None
        return torch.nn.functional.softplus(self.dhalf_bucket_raw) + 1e-6

    def _current_temporal_operator_weights(self) -> Optional[torch.Tensor]:
        if (
            not self.learned_temporal_operator_coefficients
            or self.temporal_operator_weight_raw is None
        ):
            return None
        weights = torch.nn.functional.softplus(self.temporal_operator_weight_raw) + 1e-6
        return weights / torch.clamp(weights.sum(), min=1e-6)

    def _build_spatiotemporal_schedule(
        self,
        *,
        node_eigvals: torch.Tensor,
        temporal_eigvals: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_eigvals = temporal_eigvals.to(device=device, dtype=torch.float32)
        node_eigvals = node_eigvals.to(device=device, dtype=torch.float32)
        if temporal_eigvals.ndim == 1:
            lambda_st = temporal_eigvals.unsqueeze(1) * node_eigvals.unsqueeze(0)
            mean_lambda = torch.clamp(
                lambda_st.mean(),
                min=max(self.cfg.covariance_jitter, 1e-8),
            )
        elif temporal_eigvals.ndim == 2:
            lambda_st = temporal_eigvals.unsqueeze(2) * node_eigvals.unsqueeze(0).unsqueeze(0)
            mean_lambda = torch.clamp(
                lambda_st.mean(dim=(1, 2), keepdim=True),
                min=max(self.cfg.covariance_jitter, 1e-8),
            )
        else:
            raise ValueError(
                "temporal_eigvals must have rank 1 or 2, "
                f"got shape {tuple(temporal_eigvals.shape)}"
            )
        mode_rates = torch.pow(lambda_st / mean_lambda, self.cfg.beta_matrix_power)
        mode_rates = _clamp_optional(
            mode_rates,
            min_value=self.cfg.beta_matrix_min_rate,
            max_value=self.cfg.beta_matrix_max_rate,
        )
        alpha_bars = self.schedule.alpha_bars.to(device=device, dtype=torch.float32)
        if mode_rates.ndim == 2:
            spectral_alpha_bars = torch.pow(alpha_bars[:, None, None], mode_rates.unsqueeze(0))
        else:
            spectral_alpha_bars = torch.pow(alpha_bars[:, None, None, None], mode_rates.unsqueeze(0))
        return mode_rates, spectral_alpha_bars

    def _history_temporal_mode_eigvals(self, history_free: torch.Tensor) -> torch.Tensor:
        if self.temporal_history_encoder is None or self.temporal_delta_mlp is None:
            raise RuntimeError("History-conditioned temporal anisotropy is not initialized.")
        hist = history_free.to(dtype=torch.float32)
        hist_tokens = self.temporal_history_encoder(hist)
        hist_summary = hist_tokens.mean(dim=1)
        delta = self.temporal_delta_mlp(hist_summary)
        delta_max_abs = float(self.cfg.temporal_anisotropy_delta_max_abs)
        if delta_max_abs > 0.0:
            delta = torch.clamp(delta, min=-delta_max_abs, max=delta_max_abs)
        base = self.temporal_mode_eigvals.to(device=delta.device, dtype=delta.dtype).unsqueeze(0)
        return base * torch.exp(delta)

    def _dynamic_spectral_state(
        self,
        history_free: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if (
            not self.learnable_dhalf
            and not self.temporal_anisotropy_learned_from_history
            and not self.learned_temporal_operator_coefficients
        ):
            return None
        if self.cov_builder is None:
            eigvals = self.beta_mode_eigvals
            eigvecs = self.beta_basis
            eigvecs_t = self.beta_basis_t
        else:
            bucket_variances = self._current_bucket_variances()
            empirical_feature_covariance = self._empirical_feature_covariance
            if empirical_feature_covariance is not None and bucket_variances is not None:
                empirical_feature_covariance = empirical_feature_covariance.to(
                    device=bucket_variances.device,
                    dtype=torch.float32,
                )
            cov = self.cov_builder.build_feature_covariance(
                empirical_feature_covariance=empirical_feature_covariance,
                bucket_variances=bucket_variances,
            ).to(torch.float32)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=max(self.cfg.covariance_jitter, 1e-8))
            eigvecs_t = eigvecs.t()

        temporal_basis = self.temporal_basis
        temporal_basis_t = self.temporal_basis_t
        if self.learned_temporal_operator_coefficients:
            weights = self._current_temporal_operator_weights()
            if weights is None:
                raise RuntimeError("Learned temporal operator coefficients are not initialized.")
            temporal_eigvals = _build_frequency_operator_eigvals(
                length=self.cfg.pred_length,
                operator_type=self.temporal_operator_type,
                q=self.cfg.temporal_anisotropy_q,
                velocity_weight=weights[0],
                acceleration_weight=weights[1],
                jerk_weight=weights[2],
            )
        elif self.temporal_anisotropy_learned_from_history:
            if history_free is None:
                raise ValueError("history_free is required for history-conditioned temporal anisotropy.")
            temporal_eigvals = self._history_temporal_mode_eigvals(history_free)
        else:
            temporal_eigvals = self.temporal_mode_eigvals

        _, spectral_alpha_bars = self._build_spatiotemporal_schedule(
            node_eigvals=eigvals,
            temporal_eigvals=temporal_eigvals,
            device=eigvals.device,
        )
        spectral_alpha_bars = torch.clamp(spectral_alpha_bars, min=1e-8, max=1.0)
        return eigvecs, eigvecs_t, spectral_alpha_bars, temporal_basis, temporal_basis_t

    def _select_free_joints(self, x: torch.Tensor, T: int, name: str) -> torch.Tensor:
        _assert_shape(x, (None, T, self.cfg.num_nodes, 3), name)
        return x[:, :, self.free_indices, :]

    def _flatten_free(self, x: torch.Tensor, T: int, name: str) -> torch.Tensor:
        x_free = self._select_free_joints(x, T, name)
        return x_free.reshape(x_free.size(0), T, -1)

    def _future_to_frequency_free(self, x: torch.Tensor, name: str) -> torch.Tensor:
        x_free = self._flatten_free(x, self.cfg.pred_length, name)
        future_dct = self.future_dct.to(device=x_free.device, dtype=x_free.dtype)
        return torch.matmul(future_dct, x_free)

    def _frequency_free_to_future_free(self, x_freq: torch.Tensor) -> torch.Tensor:
        _assert_shape(x_freq, (None, self.cfg.pred_length, self.free_feature_dim), "x_freq")
        future_idct = self.future_idct.to(device=x_freq.device, dtype=x_freq.dtype)
        return torch.matmul(future_idct, x_freq)

    def frequency_full_to_time_full(self, x_freq_full: torch.Tensor) -> torch.Tensor:
        _assert_shape(
            x_freq_full,
            (None, self.cfg.pred_length, self.cfg.num_nodes, 3),
            "x_freq_full",
        )
        B = x_freq_full.size(0)
        x_freq_flat = x_freq_full.reshape(B, self.cfg.pred_length, -1)
        future_idct = self.future_idct.to(device=x_freq_flat.device, dtype=x_freq_flat.dtype)
        x_flat = torch.matmul(future_idct, x_freq_flat)
        return x_flat.reshape(B, self.cfg.pred_length, self.cfg.num_nodes, 3)

    def _coarse_frequency_free(
        self,
        coarse_future_full: torch.Tensor,
        coarse_future_frequency_full: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if coarse_future_frequency_full is not None:
            return self._flatten_free(
                coarse_future_frequency_full,
                self.cfg.pred_length,
                "coarse_future_frequency_full",
            )
        return self._future_to_frequency_free(coarse_future_full, "coarse_future_full")

    def _restore_full_from_free(self, x_free_flat: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
        B = x_free_flat.size(0)
        _assert_shape(x_free_flat, (B, T, self.free_feature_dim), "x_free_flat")
        x_free = x_free_flat.reshape(B, T, self.num_free_nodes, 3)
        full = torch.zeros(B, T, self.cfg.num_nodes, 3, device=x_free.device, dtype=dtype)
        full[:, :, self.free_indices, :] = x_free
        return full

    def _spectral_schedule_rows(
        self,
        timesteps: torch.Tensor,
        dtype: torch.dtype,
        spectral_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if spectral_state is None:
            spectral_state = self._dynamic_spectral_state()
        alpha_bars_source = self.spectral_alpha_bars if spectral_state is None else spectral_state[2]
        alpha_bars = _index_schedule_rows(alpha_bars_source, timesteps, dtype)
        return alpha_bars, torch.sqrt(alpha_bars), torch.sqrt(torch.clamp(1.0 - alpha_bars, min=0.0))

    def _to_spectral(
        self,
        x: torch.Tensor,
        spectral_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if spectral_state is None:
            spectral_state = self._dynamic_spectral_state()
        basis_source = self.beta_basis if spectral_state is None else spectral_state[0]
        basis = basis_source.to(device=x.device, dtype=x.dtype)
        temporal_basis_t_source = self.temporal_basis_t if spectral_state is None else spectral_state[4]
        temporal_basis_t = temporal_basis_t_source.to(device=x.device, dtype=x.dtype)
        return torch.matmul(temporal_basis_t, torch.matmul(x, basis))

    def _from_spectral(
        self,
        z: torch.Tensor,
        spectral_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if spectral_state is None:
            spectral_state = self._dynamic_spectral_state()
        basis_t_source = self.beta_basis_t if spectral_state is None else spectral_state[1]
        basis_t = basis_t_source.to(device=z.device, dtype=z.dtype)
        temporal_basis_source = self.temporal_basis if spectral_state is None else spectral_state[3]
        temporal_basis = temporal_basis_source.to(device=z.device, dtype=z.dtype)
        return torch.matmul(temporal_basis, torch.matmul(z, basis_t))

    def training_loss(
        self,
        *,
        residual_gt: torch.Tensor,
        history_full: torch.Tensor,
        coarse_future_full: torch.Tensor,
        future_gt_full: Optional[torch.Tensor] = None,
        coarse_future_frequency_full: Optional[torch.Tensor] = None,
        mamp_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = residual_gt.size(0)
        _assert_shape(residual_gt, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "residual_gt")
        _assert_shape(history_full, (B, self.cfg.input_length, self.cfg.num_nodes, 3), "history_full")
        _assert_shape(coarse_future_full, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "coarse_future_full")

        history_free = self._flatten_free(history_full, self.cfg.input_length, "history_full")
        coarse_free = self._coarse_frequency_free(coarse_future_full, coarse_future_frequency_full)
        if future_gt_full is not None:
            _assert_shape(future_gt_full, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "future_gt_full")
            x0 = self._future_to_frequency_free(future_gt_full, "future_gt_full") - coarse_free
        else:
            x0 = self._future_to_frequency_free(residual_gt, "residual_gt")
        x0 = x0.float()
        device = x0.device
        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=device, dtype=torch.long)
        spectral_state = self._dynamic_spectral_state(history_free=history_free)
        x0_z = self._to_spectral(x0, spectral_state=spectral_state)
        eps_z = torch.randn_like(x0_z)
        _, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = self._spectral_schedule_rows(
            t,
            x0.dtype,
            spectral_state=spectral_state,
        )
        sqrt_ab = sqrt_alpha_bars
        sqrt_omb = sqrt_one_minus_alpha_bars
        x_t_z = sqrt_ab * x0_z + sqrt_omb * eps_z
        v_z = sqrt_ab * eps_z - sqrt_omb * x0_z
        x_t = self._from_spectral(x_t_z, spectral_state=spectral_state)
        v = self._from_spectral(v_z, spectral_state=spectral_state)

        v_hat = self.denoiser(
            x_t.to(dtype=history_free.dtype),
            t,
            history_free,
            coarse_free,
            mamp_feat=mamp_feat,
        ).float()

        err = v_hat - v
        return torch.mean(err ** 2)

    @torch.no_grad()
    def sample_ddim(
        self,
        *,
        history_full: torch.Tensor,
        coarse_future_full: torch.Tensor,
        coarse_future_frequency_full: Optional[torch.Tensor] = None,
        mamp_feat: Optional[torch.Tensor] = None,
        seed: int = 0,
        steps: Optional[int] = None,
        eta: float = 0.0,
        return_score: bool = False,
    ) -> torch.Tensor:
        if steps is None:
            steps = self.cfg.ddim_steps
        steps = int(steps)
        if steps <= 0:
            raise ValueError("DDIM steps must be positive")

        B = history_full.size(0)
        device = history_full.device
        _assert_shape(history_full, (B, self.cfg.input_length, self.cfg.num_nodes, 3), "history_full")
        _assert_shape(coarse_future_full, (B, self.cfg.pred_length, self.cfg.num_nodes, 3), "coarse_future_full")

        history_free = self._flatten_free(history_full, self.cfg.input_length, "history_full")
        coarse_free = self._coarse_frequency_free(coarse_future_full, coarse_future_frequency_full)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        T = self.cfg.diffusion_steps
        t_seq = _make_ddim_timesteps(T, steps, device=device)
        clip = float(self.cfg.x0_clip)
        score = torch.zeros(B, device=device, dtype=torch.float32) if return_score else None
        x = torch.randn(
            (B, self.cfg.pred_length, self.free_feature_dim),
            device=device, dtype=torch.float32, generator=gen,
        )
        spectral_state = self._dynamic_spectral_state(history_free=history_free)
        alpha_init, _, sqrt_omb_init = self._spectral_schedule_rows(
            torch.full((B,), int(t_seq[0].item()), device=device, dtype=torch.long),
            x.dtype,
            spectral_state=spectral_state,
        )
        del alpha_init
        x_z = sqrt_omb_init * x

        for idx in range(t_seq.numel()):
            t_int = int(t_seq[idx].item())
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

            x = self._from_spectral(x_z, spectral_state=spectral_state)
            x_input = x.to(dtype=history_free.dtype)

            v_hat = self.denoiser(
                x_input, t_batch, history_free, coarse_free, mamp_feat=mamp_feat,
            ).float()
            _, sqrt_alpha_bars_t, sqrt_one_minus_alpha_bars_t = self._spectral_schedule_rows(
                t_batch,
                x.dtype,
                spectral_state=spectral_state,
            )
            sqrt_ab_t = sqrt_alpha_bars_t
            sqrt_omb_t = sqrt_one_minus_alpha_bars_t
            v_hat_z = self._to_spectral(v_hat, spectral_state=spectral_state)
            x0_z = sqrt_ab_t * x_z - sqrt_omb_t * v_hat_z
            x0 = self._from_spectral(x0_z, spectral_state=spectral_state)
            if clip > 0.0:
                x0 = torch.clamp(x0, -clip, clip)
                x0_z = self._to_spectral(x0, spectral_state=spectral_state)
            eps_hat_z = sqrt_omb_t * x_z + sqrt_ab_t * v_hat_z
            if score is not None:
                score = score + eps_hat_z.float().pow(2).sum(dim=(1, 2))

            if idx == t_seq.numel() - 1:
                alpha_prev = torch.ones(
                    B,
                    self.cfg.pred_length,
                    self.free_feature_dim,
                    device=device,
                    dtype=x.dtype,
                )
            else:
                t_prev = torch.full((B,), int(t_seq[idx + 1].item()), device=device, dtype=torch.long)
                alpha_prev, _, _ = self._spectral_schedule_rows(t_prev, x.dtype, spectral_state=spectral_state)
            sqrt_ab_prev = torch.sqrt(alpha_prev)
            sqrt_omb_prev = torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0))

            if eta != 0.0:
                alpha_t = torch.clamp(sqrt_ab_t ** 2, min=1e-8, max=1.0)
                sigma = eta * torch.sqrt(
                    torch.clamp(
                        (1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev),
                        min=0.0,
                    )
                )
                dir_coeff = torch.sqrt(torch.clamp(sqrt_omb_prev ** 2 - sigma ** 2, min=0.0))
                noise_z = torch.randn(x_z.shape, device=x_z.device, dtype=x_z.dtype, generator=gen)
                x_z = sqrt_ab_prev * x0_z + dir_coeff * eps_hat_z + sigma * noise_z
            else:
                x_z = sqrt_ab_prev * x0_z + sqrt_omb_prev * eps_hat_z

        out_frequency_free = self._from_spectral(x_z, spectral_state=spectral_state)
        out_free = self._frequency_free_to_future_free(out_frequency_free).to(dtype=history_full.dtype)
        out_full = self._restore_full_from_free(out_free, self.cfg.pred_length, history_full.dtype)
        if score is None:
            return out_full
        return out_full, score


# ----------------------------- Wrapper model ----------------------------------

class CardForecaster(nn.Module):
    def __init__(self, cfg: CardConfig, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.coarse = CoarseDCTForecaster(cfg)
        self.diffusion = TimeResidualConditionalDiffusion(cfg, metadata=metadata)
        if cfg.freeze_coarse:
            for p in self.coarse.parameters():
                p.requires_grad = False

    def _zero_coarse_future(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        return torch.zeros(
            history.size(0),
            self.cfg.pred_length,
            self.cfg.num_nodes,
            3,
            device=history.device,
            dtype=history.dtype,
        )

    def _zero_coarse_frequency(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        return torch.zeros(
            history.size(0),
            self.cfg.pred_length,
            self.cfg.num_nodes,
            3,
            device=history.device,
            dtype=history.dtype,
        )

    def _coarse_time_and_frequency(
        self,
        history: torch.Tensor,
        coarse_future: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cfg.diffusion_only:
            coarse_freq = self._zero_coarse_frequency(history)
            coarse_time = self._zero_coarse_future(history)
            return coarse_time, coarse_freq
        if coarse_future is not None:
            _assert_shape(
                coarse_future,
                (history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3),
                "coarse_future",
            )
            return coarse_future, None
        coarse_freq_flat = self.coarse.predict_future_coeffs(history)
        coarse_freq = coarse_freq_flat.reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )
        coarse_time = self.diffusion.frequency_full_to_time_full(coarse_freq)
        return coarse_time, coarse_freq

    def diffusion_loss(
        self,
        history: torch.Tensor,
        future_gt: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
        coarse_future: Optional[torch.Tensor] = None,
        allow_coarse_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        _assert_shape(future_gt, (None, self.cfg.pred_length, self.cfg.num_nodes, 3), "future_gt")

        coarse_future, coarse_frequency = self._coarse_time_and_frequency(history, coarse_future)
        if allow_coarse_grad:
            coarse_cond = coarse_future if not self.cfg.stopgrad_coarse_condition else coarse_future.detach()
            coarse_freq_cond = (
                coarse_frequency if coarse_frequency is not None and not self.cfg.stopgrad_coarse_condition
                else coarse_frequency.detach() if coarse_frequency is not None
                else None
            )
            residual_gt = future_gt - coarse_future
        else:
            coarse_cond = coarse_future.detach() if self.cfg.stopgrad_coarse_condition else coarse_future
            coarse_freq_cond = coarse_frequency.detach() if coarse_frequency is not None else None
            residual_gt = future_gt - coarse_future.detach()

        loss = self.diffusion.training_loss(
            residual_gt=residual_gt,
            history_full=history,
            coarse_future_full=coarse_cond,
            future_gt_full=future_gt,
            coarse_future_frequency_full=coarse_freq_cond,
            mamp_feat=mamp_feat,
        )
        return loss, coarse_future

    @torch.no_grad()
    def predict(
        self,
        history: torch.Tensor,
        mamp_feat: Optional[torch.Tensor] = None,
        coarse_future: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        seed: int = 0,
        return_score: bool = False,
    ) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        coarse_future, coarse_frequency = self._coarse_time_and_frequency(history, coarse_future)

        coarse_cond = coarse_future.detach() if self.cfg.stopgrad_coarse_condition else coarse_future
        coarse_freq_cond = coarse_frequency.detach() if coarse_frequency is not None else None
        residual_pred = self.diffusion.sample_ddim(
            history_full=history,
            coarse_future_full=coarse_cond,
            coarse_future_frequency_full=coarse_freq_cond,
            mamp_feat=mamp_feat,
            seed=seed,
            steps=self.cfg.ddim_steps,
            eta=0.0 if deterministic else 1.0,
            return_score=return_score,
        )

        if return_score:
            residual_pred, score = residual_pred
        future = coarse_future + residual_pred
        if 0 <= int(self.diffusion.wrist_index) < self.cfg.num_nodes:
            future[:, :, int(self.diffusion.wrist_index), :] = 0.0
        if return_score:
            return future, score
        return future

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.cfg.diffusion_only:
                return self._zero_coarse_future(history)
            return self.coarse(history)
        return self.predict(history, deterministic=True, seed=0)
