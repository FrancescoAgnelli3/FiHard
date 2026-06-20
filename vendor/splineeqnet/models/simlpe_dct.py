"""siMLPe baseline with temporal DCT pre/post-processing.

This module follows the public siMLPe baseline implementation that operates
in the DCT domain before applying the temporal MLP. It mirrors the custom
Assembly forecaster but preserves the DCT projection used in the original
siMLPe setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
from typing import Literal

import torch
from torch import nn


class _SpatialNorm(nn.Module):
    """LayerNorm-style normalization over the spatial/channel axis."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        y = (x - mean) * inv_std
        return y * self.alpha + self.beta


class _TemporalNorm(nn.Module):
    """LayerNorm-style normalization over the temporal axis."""

    def __init__(self, seq_len: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, seq_len))
        self.beta = nn.Parameter(torch.zeros(1, 1, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        y = (x - mean) * inv_std
        return y * self.alpha + self.beta


class _SpatialFC(nn.Module):
    """Linear projection applied across the feature/channel axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> transpose to operate over channels
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        return x.transpose(1, 2)


class _TemporalFC(nn.Module):
    """Linear projection applied across the temporal axis."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear acts on the last dimension, which is the temporal axis here.
        return self.proj(x)


class _MLPBlock(nn.Module):
    """Residual block used by siMLPe."""

    def __init__(
        self,
        dim: int,
        seq_len: int,
        *,
        use_norm: bool = True,
        use_spatial_fc: bool = False,
        norm_axis: Literal["spatial", "temporal", "all"] = "spatial",
    ) -> None:
        super().__init__()
        if use_spatial_fc:
            self.fc = _SpatialFC(dim)
        else:
            self.fc = _TemporalFC(seq_len)

        if use_norm:
            if norm_axis == "spatial":
                self.norm = _SpatialNorm(dim)
            elif norm_axis == "temporal":
                self.norm = _TemporalNorm(seq_len)
            elif norm_axis == "all":
                self.norm = nn.LayerNorm([dim, seq_len])
            else:
                raise ValueError(f"Unknown norm_axis={norm_axis}")
        else:
            self.norm = nn.Identity()

        # Follow the initialization from the public siMLPe implementation.
        if hasattr(self.fc, "proj"):
            nn.init.xavier_uniform_(self.fc.proj.weight, gain=1e-8)
            nn.init.constant_(self.fc.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc(x)
        out = self.norm(out)
        return residual + out


class _TransMLP(nn.Module):
    """Stack of `_MLPBlock`s mirroring the siMLPe backbone."""

    def __init__(
        self,
        dim: int,
        seq_len: int,
        *,
        use_norm: bool,
        use_spatial_fc: bool,
        num_layers: int,
        norm_axis: Literal["spatial", "temporal", "all"],
        mix_spatial_temporal: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive for TransMLP.")

        if mix_spatial_temporal:
            layer_spatial_flags = [(idx % 2) == 1 for idx in range(num_layers)]
        else:
            layer_spatial_flags = [use_spatial_fc] * num_layers

        layers = [
            _MLPBlock(
                dim,
                seq_len,
                use_norm=use_norm,
                use_spatial_fc=use_spatial_flag,
                norm_axis=norm_axis,
            )
            for use_spatial_flag in layer_spatial_flags
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_dct_matrix(total_len: int) -> torch.Tensor:
    """Construct an orthonormal DCT-II projection matrix."""
    if total_len <= 0:
        raise ValueError("Sequence length for DCT must be positive.")
    # Rows correspond to DCT frequency indices, so the first row is the DC mode.
    k = torch.arange(total_len, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(total_len, dtype=torch.float32).unsqueeze(0)
    scale = math.sqrt(2.0 / total_len)
    mat = scale * torch.cos(math.pi * (n + 0.5) * k / float(total_len))
    mat[0, :] = math.sqrt(1.0 / total_len)
    return mat

@dataclass
class SiMLPeConfig:
    """Configuration for the siMLPe competitor."""

    input_length: int
    pred_length: int
    num_nodes: int
    hidden_dim: int
    num_layers: int
    use_norm: bool = True
    use_spatial_fc_only: bool = False
    mix_spatial_temporal: bool = False
    norm_axis: Literal["spatial", "temporal", "all"] = "spatial"
    add_last_offset: bool = True

    @property
    def total_length(self) -> int:
        return self.input_length + self.pred_length

    @property
    def feature_dim(self) -> int:
        return self.num_nodes * 3


@dataclass
class SiMLPeDCTConfig(SiMLPeConfig):
    """siMLPe configuration with optional DCT coefficient trimming."""

    dct_components: Optional[int] = None

    @property
    def dct_length(self) -> int:
        keep = self.dct_components
        total = self.total_length
        if keep is None:
            return total
        return max(1, min(int(keep), total))

class SiMLPeBackbone(nn.Module):
    """Feature-space implementation of the siMLPe temporal MLP."""

    def __init__(self, cfg: SiMLPeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc_in = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.mlp = _TransMLP(
            cfg.hidden_dim,
            cfg.total_length,
            use_norm=cfg.use_norm,
            use_spatial_fc=cfg.use_spatial_fc_only,
            num_layers=cfg.num_layers,
            norm_axis=cfg.norm_axis,
            mix_spatial_temporal=cfg.mix_spatial_temporal,
        )
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.feature_dim)

        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T_total, F)
        x = self.fc_in(seq)
        x = x.transpose(1, 2).contiguous()  # (B, hidden_dim, T_total)
        x = self.mlp(x)
        x = x.transpose(1, 2).contiguous()
        return self.fc_out(x)



class SiMLPeDCTForecaster(nn.Module):
    """End-to-end forecaster that applies siMLPe in the DCT domain."""

    def __init__(self, cfg: SiMLPeDCTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        seq_len = cfg.total_length

        dct = _build_dct_matrix(seq_len)
        idct = dct.t()
        keep = cfg.dct_length
        # Keep low-frequency bands if requested; otherwise use the full matrix.
        self.register_buffer("_dct", dct[:keep, :], persistent=False)
        self.register_buffer("_idct", idct[:, :keep], persistent=False)
        self.full_seq_len = seq_len
        self.dct_len = keep

        self.backbone = SiMLPeBackbone(cfg)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict future 3D joint coordinates from observed history.

        Args:
            history: Tensor with shape (B, T_in, N, 3).

        Returns:
            Tensor with shape (B, T_out, N, 3).
        """
        if history.ndim != 4:
            raise ValueError("history must be shaped (B, T_in, N, 3)")
        batch, t_in, nodes, coord = history.shape
        if t_in != self.cfg.input_length:
            raise ValueError(f"Expected {self.cfg.input_length} input frames, got {t_in}.")
        if nodes != self.cfg.num_nodes or coord != 3:
            raise ValueError("Input history must have N=num_nodes and coordinate dim=3.")

        flat = history.reshape(batch, t_in, -1)
        last_frame = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
        padded = torch.cat([flat, last_frame], dim=1)  # (B, T_total, F)

        dct = self._dct.unsqueeze(0).to(dtype=padded.dtype, device=padded.device)
        dct_coeffs = torch.matmul(dct, padded)

        # Zero out discarded bands (if any) while keeping tensor width for the backbone.
        if self.dct_len < self.full_seq_len:
            padded_coeffs = torch.zeros(
                batch,
                self.full_seq_len,
                padded.size(-1),
                dtype=dct_coeffs.dtype,
                device=dct_coeffs.device,
            )
            padded_coeffs[:, : self.dct_len, :] = dct_coeffs
            proc_in = padded_coeffs
        else:
            proc_in = dct_coeffs

        proc_out = self.backbone(proc_in)

        if self.dct_len < self.full_seq_len:
            proc_out = proc_out[:, : self.dct_len, :]

        idct = self._idct.unsqueeze(0).to(dtype=proc_out.dtype, device=proc_out.device)
        recon = torch.matmul(idct, proc_out)

        future = recon[:, -self.cfg.pred_length :, :]
        future = future.reshape(batch, self.cfg.pred_length, nodes, coord)

        if self.cfg.add_last_offset:
            future = future + history[:, -1:, :, :]
        return future


__all__ = ["SiMLPeDCTConfig", "SiMLPeDCTForecaster"]
