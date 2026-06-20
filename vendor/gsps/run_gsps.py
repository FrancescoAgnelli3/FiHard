#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch import nn

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.evaluation import save_eval_samples_npz
from common.metrics import humanmac_metrics_prefixed
from models.motion_pred_ours import get_model
from utils.util import absolute2relative_torch, get_dct_matrix


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_dict(obj: object) -> Dict[str, object]:
    return dict(obj) if isinstance(obj, dict) else {}


def _import_splineeqnet_modules(splineeqnet_root: Path):
    root = splineeqnet_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"SplineEqNet root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for module_name in ("config", "data", "datasets"):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        mod_path = Path(getattr(mod, "__file__", "")).resolve()
        if not str(mod_path).startswith(str(root)):
            del sys.modules[module_name]

    config_mod = importlib.import_module("config")
    data_mod = importlib.import_module("data")
    return config_mod, data_mod


def _extract_xyz(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected [B,T,N,C], got shape={tuple(x.shape)}")
    c = int(x.shape[-1])
    if c == 3:
        return x
    if c >= 7:
        return x[..., 4:7]
    if c > 3:
        return x[..., -3:]
    raise ValueError(f"Cannot extract xyz from channel dim={c}")


def _prepare_batch(
    batch: Sequence[torch.Tensor],
    *,
    t_his: int,
    t_pred: int,
    n_pre: int,
    dct_m: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise RuntimeError("Expected batch tuple/list with at least (input, output)")

    inp = batch[0].to(device=device, dtype=torch.float32)
    out = batch[1].to(device=device, dtype=torch.float32)
    norm_factor = (
        batch[2].to(device=device, dtype=torch.float32).view(-1)
        if len(batch) > 2
        else torch.ones(inp.shape[0], device=device, dtype=torch.float32)
    )

    inp_xyz = _extract_xyz(inp)
    out_xyz = _extract_xyz(out)

    # Keep legacy root handling for compatibility with the existing integrated GSPS runner.
    inp_nr = inp_xyz[:, :, 1:, :]
    out_nr = out_xyz[:, :, 1:, :]

    bsz, _, n_nr, _ = inp_nr.shape
    t_total = int(t_his + t_pred)

    history_padded = torch.cat([inp_nr, inp_nr[:, -1:, :, :].expand(-1, t_pred, -1, -1)], dim=1)
    if history_padded.shape[1] != t_total:
        raise RuntimeError(f"Unexpected padded length: got {history_padded.shape[1]}, expected {t_total}")

    inp_flat = history_padded.reshape(bsz, t_total, -1)
    inp_dct = torch.matmul(dct_m[:n_pre], inp_flat).transpose(1, 2).reshape(bsz, n_nr, 3, n_pre).reshape(bsz, n_nr, -1)

    gt_future_flat = out_nr.reshape(bsz, t_pred, -1)
    gt_total_flat = torch.cat([inp_nr, out_nr], dim=1).reshape(bsz, t_total, -1)
    conditioning_context_nr = inp_nr
    return inp_dct, gt_future_flat, gt_total_flat, conditioning_context_nr, norm_factor


def _sample_predictions(
    model: nn.Module,
    *,
    inp_dct: torch.Tensor,
    idct_m: torch.Tensor,
    t_his: int,
    t_pred: int,
    nk: int,
    nz: int,
    n_parts: int,
    seed: int,
    cartesian: bool,
    return_total: bool,
) -> torch.Tensor:
    bsz, n_nr, _ = inp_dct.shape
    t_total = int(t_his + t_pred)
    gen = torch.Generator(device=inp_dct.device)
    gen.manual_seed(int(seed))

    if cartesian:
        total_k = int(nk ** n_parts)
        if total_k <= 0:
            raise RuntimeError("Invalid GSPS cartesian sample count")

        inp_rep = inp_dct.unsqueeze(1).expand(-1, total_k, -1, -1).reshape(bsz * total_k, n_nr, -1)

        # Build cartesian products of per-part latent indices, matching original GSPS training protocol.
        base_z = torch.randn(bsz, nk, n_parts, nz, device=inp_dct.device, dtype=inp_dct.dtype, generator=gen)
        combos = list(itertools.product(range(nk), repeat=n_parts))
        if len(combos) != total_k:
            raise RuntimeError("GSPS latent cartesian product construction failed")
        z = torch.empty(bsz, total_k, n_parts, nz, device=inp_dct.device, dtype=inp_dct.dtype)
        for ci, combo in enumerate(combos):
            for pi, ki in enumerate(combo):
                z[:, ci, pi] = base_z[:, ki, pi]
        z = z.reshape(bsz * total_k, n_parts, nz)
    else:
        total_k = int(nk)
        inp_rep = inp_dct.unsqueeze(1).expand(-1, total_k, -1, -1).reshape(bsz * total_k, n_nr, -1)
        z = torch.randn(bsz * total_k, n_parts, nz, device=inp_dct.device, dtype=inp_dct.dtype, generator=gen)

    xt = model(inp_rep, z)
    n_pre = xt.shape[-1] // 3
    xt = xt.reshape(bsz * total_k, n_nr, 3, n_pre).reshape(bsz * total_k, n_nr * 3, n_pre).transpose(1, 2)

    traj_est = torch.matmul(idct_m[:, :n_pre], xt)
    if traj_est.shape[1] != t_total:
        raise RuntimeError(f"Unexpected reconstructed length={traj_est.shape[1]}, expected {t_total}")

    if return_total:
        return traj_est.reshape(bsz, total_k, t_total, -1)
    pred_future = traj_est[:, t_his:, :]
    return pred_future.reshape(bsz, total_k, t_pred, -1)


def _build_parents_from_links(n_nodes: int, links: Sequence[Tuple[int, int]], root: int = 0) -> List[int]:
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for e in links:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue
        a = int(e[0])
        b = int(e[1])
        if 0 <= a < n_nodes and 0 <= b < n_nodes and a != b:
            adj[a].append(b)
            adj[b].append(a)

    parents = [-1] * n_nodes
    seen = [False] * n_nodes
    queue = [int(root)]
    if not (0 <= int(root) < n_nodes):
        queue = [0]
    seen[queue[0]] = True
    parents[queue[0]] = -1

    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if seen[nb]:
                continue
            seen[nb] = True
            parents[nb] = cur
            queue.append(nb)

    # Fallback for disconnected nodes.
    for i in range(n_nodes):
        if not seen[i]:
            parents[i] = 0 if i != 0 else -1

    return parents


def _relative_pose_features(pose_nr: torch.Tensor, parents_full: Sequence[int]) -> torch.Tensor:
    # pose_nr: [B, N_nr, 3], returns [B, N_nr*3] as relative vectors wrt parent tree.
    if pose_nr.ndim != 3 or pose_nr.shape[-1] != 3:
        raise ValueError(f"Expected pose_nr [B,N,3], got {tuple(pose_nr.shape)}")
    bsz, n_nr, _ = pose_nr.shape
    full = torch.zeros(bsz, n_nr + 1, 3, dtype=pose_nr.dtype, device=pose_nr.device)
    full[:, 1:] = pose_nr
    rel = absolute2relative_torch(full, parents=list(parents_full))
    if rel.shape[1] != n_nr:
        raise RuntimeError(f"Unexpected relative pose joints: got {rel.shape[1]}, expected {n_nr}")
    return rel.reshape(bsz, n_nr * 3)


def _build_multimodal_candidates(
    gt_future: torch.Tensor,
    gt_total: torch.Tensor,
    *,
    t_his: int,
    n_modality: int,
) -> torch.Tensor:
    # Build in-batch multimodal futures by nearest history neighbors.
    bsz = int(gt_future.shape[0])
    if bsz <= 1 or n_modality <= 1:
        return gt_future.unsqueeze(1)

    hist = gt_total[:, :t_his, :].reshape(bsz, -1)
    dist = torch.cdist(hist, hist, p=2)
    k = min(int(n_modality), bsz)
    nn_idx = dist.topk(k=k, largest=False, dim=1).indices  # [B,k]
    idx_exp = nn_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, gt_future.shape[1], gt_future.shape[2])
    mm = torch.gather(gt_future.unsqueeze(1).expand(-1, bsz, -1, -1), dim=1, index=idx_exp)
    return mm


def _part_feature_indices(parts: Sequence[Sequence[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for p in parts:
        idx: List[int] = []
        for j in p:
            jj = int(j)
            idx.extend([jj * 3, jj * 3 + 1, jj * 3 + 2])
        out.append(idx)
    return out


def _joint_diversity_loss(
    pred_future: torch.Tensor,
    *,
    nk: int,
    n_parts: int,
    parts: Sequence[Sequence[int]],
    alpha: float,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pred_future: [B, Ktot, T, F]
    device = pred_future.device
    dtype = pred_future.dtype
    if n_parts != 2:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, z

    bsz, k_tot, t_pred, _ = pred_future.shape
    if k_tot != nk ** n_parts:
        z = torch.zeros((), device=device, dtype=dtype)
        return z, z, z

    part_idx = _part_feature_indices(parts)
    pred = pred_future.view(bsz, nk, nk, t_pred, -1)

    mask = torch.triu(torch.ones(nk, nk, device=device, dtype=torch.bool), diagonal=1)

    yt0 = pred[:, :, 0, :, :][..., part_idx[0]].reshape(bsz, nk, -1)
    pd0 = torch.cdist(yt0, yt0, p=1)
    loss_div_l = torch.exp(-pd0[:, mask] / float(alpha)).mean()

    yt1 = pred[..., part_idx[1]].reshape(bsz * nk, nk, -1)
    pd1 = torch.cdist(yt1, yt1, p=1)
    loss_div_u = torch.exp(-pd1[:, mask] / float(beta)).mean()

    with torch.no_grad():
        mask_all = torch.triu(torch.ones(k_tot, k_tot, device=device, dtype=torch.bool), diagonal=1)
        yf = pred_future.reshape(bsz, k_tot, -1)
        pd_all = torch.cdist(yf, yf, p=2)
        div = pd_all[:, mask_all].mean()

    return loss_div_l, loss_div_u, div


def _recon_losses(pred_future: torch.Tensor, gt_future: torch.Tensor, gt_multi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # pred_future [B,K,T,F], gt_future [B,T,F], gt_multi [B,M,T,F]
    diff = pred_future - gt_future.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)  # [B,K]
    loss_recon = dist.min(dim=1).values.mean()
    with torch.no_grad():
        ade = torch.norm(diff, dim=-1).mean(dim=-1).min(dim=1).values.mean()

    diff_mm = pred_future.unsqueeze(2) - gt_multi.unsqueeze(1)  # [B,K,M,T,F]
    dist_mm = diff_mm.pow(2).sum(dim=-1).sum(dim=-1)  # [B,K,M]
    mask = gt_multi.abs().sum(dim=-1).sum(dim=-1) > 1e-6  # [B,M]
    best_mm = dist_mm.min(dim=1).values
    if torch.any(mask):
        loss_recon_mm = best_mm[mask].mean()
    else:
        loss_recon_mm = torch.zeros_like(loss_recon)
    if torch.isnan(loss_recon_mm):
        loss_recon_mm = torch.zeros_like(loss_recon)
    return loss_recon, loss_recon_mm, ade


def _limb_loss(pred_total: torch.Tensor, gt_total: torch.Tensor, parents_nr: Sequence[int]) -> torch.Tensor:
    # pred_total [B,K,T,N*3], gt_total [B,T,N*3]
    bsz, k_tot, t_total, feat = pred_total.shape
    n_nr = feat // 3
    pred_xyz = pred_total.reshape(bsz, k_tot, t_total, n_nr, 3)
    gt_xyz = gt_total.reshape(bsz, t_total, n_nr, 3)

    edges = [(j, int(parents_nr[j])) for j in range(n_nr) if int(parents_nr[j]) >= 0 and int(parents_nr[j]) < n_nr]
    if not edges:
        return torch.zeros((), device=pred_total.device, dtype=pred_total.dtype)

    child = torch.tensor([e[0] for e in edges], device=pred_total.device, dtype=torch.long)
    parent = torch.tensor([e[1] for e in edges], device=pred_total.device, dtype=torch.long)

    limb_gt = torch.norm(gt_xyz[:, 0:1, child] - gt_xyz[:, 0:1, parent], dim=-1).squeeze(1)  # [B,E]
    limb_pred = torch.norm(pred_xyz[:, :, :, child] - pred_xyz[:, :, :, parent], dim=-1)  # [B,K,T,E]
    return ((limb_pred - limb_gt[:, None, None, :]) ** 2).sum(dim=-1).mean()


def _history_loss(pred_total: torch.Tensor, gt_total: torch.Tensor, t_his: int) -> torch.Tensor:
    pred_h = pred_total[:, :, :t_his, :]
    gt_h = gt_total[:, None, :t_his, :]
    return ((pred_h - gt_h) ** 2).sum(dim=-1).mean()


def _composite_loss(
    *,
    pred_total: torch.Tensor,
    gt_total: torch.Tensor,
    gt_future: torch.Tensor,
    gt_multi: torch.Tensor,
    pose_prior: nn.Module,
    prior_dist: torch.distributions.Normal,
    parents_full: Sequence[int],
    parents_nr: Sequence[int],
    nk: int,
    n_parts: int,
    parts: Sequence[Sequence[int]],
    alphas: Sequence[float],
    lambdas: Sequence[float],
    t_his: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_future = pred_total[:, :, t_his:, :]

    loss_div_l, loss_div_u, div = _joint_diversity_loss(
        pred_future,
        nk=nk,
        n_parts=n_parts,
        parts=parts,
        alpha=float(alphas[0]),
        beta=float(alphas[1]),
    )
    loss_recon, loss_recon_mm, ade = _recon_losses(pred_future, gt_future, gt_multi)
    loss_x = _history_loss(pred_total, gt_total, t_his=t_his)
    loss_limb = _limb_loss(pred_total, gt_total, parents_nr=parents_nr)

    # Prior term over generated future poses (stride-3 subset as in original code).
    traj = pred_total[:, :, t_his:, :]
    offset = int(np.random.randint(0, 3)) if traj.shape[2] >= 3 else 0
    traj_sub = traj[:, :, offset::3, :]
    pose_nr = traj_sub.reshape(-1, traj_sub.shape[-1] // 3, 3)
    x_rel = _relative_pose_features(pose_nr, parents_full=parents_full)
    z, prior_logdet = pose_prior(x_rel)
    prior_lkh = prior_dist.log_prob(z).sum(dim=-1)

    if len(lambdas) < 9:
        raise RuntimeError("GSPS composite loss expects 9 lambda coefficients")

    loss = (
        loss_x * float(lambdas[0])
        + loss_limb * float(lambdas[1])
        + loss_div_l * float(lambdas[2])
        + loss_div_u * float(lambdas[3])
        + loss_recon * float(lambdas[4])
        + loss_recon_mm * float(lambdas[5])
        - prior_lkh.mean() * float(lambdas[6])
    )

    # Keep slot lambdas[8] for angle penalty compatibility (disabled for hand data by default).
    if float(lambdas[8]) != 0.0:
        loss = loss + torch.zeros((), device=loss.device, dtype=loss.dtype)

    log = {
        "loss": float(loss.item()),
        "loss_x": float(loss_x.item()),
        "loss_limb": float(loss_limb.item()),
        "loss_div_l": float(loss_div_l.item()),
        "loss_div_u": float(loss_div_u.item()),
        "div": float(div.item()) if torch.is_tensor(div) else float(div),
        "loss_recon": float(loss_recon.item()),
        "loss_recon_mm": float(loss_recon_mm.item()),
        "ade": float(ade.item()),
        "prior_lkh": float(prior_lkh.mean().item()),
        "prior_logdet": float(prior_logdet.mean().item()) if torch.is_tensor(prior_logdet) else float(prior_logdet),
    }
    return loss, log


def _train_prior_stage(
    *,
    pose_prior: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    parents_full: Sequence[int],
    seed: int,
    device: torch.device,
    early_stopping_enabled: bool = False,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_warmup: int = 0,
) -> None:
    if epochs <= 0:
        return

    optimizer = torch.optim.Adam(pose_prior.parameters(), lr=lr, weight_decay=weight_decay)
    prior = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    es_enabled = bool(early_stopping_enabled)
    es_patience = max(1, int(early_stopping_patience))
    es_min_delta = float(early_stopping_min_delta)
    es_warmup = max(0, int(early_stopping_warmup))
    es_best = None
    es_bad_epochs = 0

    pose_prior.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        steps = 0
        for bidx, batch in enumerate(train_loader):
            inp = _extract_xyz(batch[0].to(device=device, dtype=torch.float32))[:, :, 1:, :]
            out = _extract_xyz(batch[1].to(device=device, dtype=torch.float32))[:, :, 1:, :]
            seq = torch.cat([inp, out], dim=1)  # [B,T,N,3]

            # Random pose sample per sequence.
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + epoch * 1000003 + bidx)
            t_idx = torch.randint(low=0, high=seq.shape[1], size=(seq.shape[0],), device=device, generator=gen)
            pose_nr = seq[torch.arange(seq.shape[0], device=device), t_idx]
            x = _relative_pose_features(pose_nr, parents_full=parents_full)

            z, log_det = pose_prior(x)
            prior_likelihood = prior.log_prob(z).sum(dim=1)
            loss = -prior_likelihood.mean() - log_det.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pose_prior.parameters(), max_norm=100.0)
            optimizer.step()

            loss_sum += float(loss.item())
            steps += 1

        mean_loss = loss_sum / max(1, steps)
        print(f"[GSPS][prior][epoch {epoch + 1}/{epochs}] nll={mean_loss:.6f}")
        if es_enabled and (epoch + 1) > es_warmup and np.isfinite(mean_loss):
            improved = es_best is None or mean_loss < (float(es_best) - es_min_delta)
            if improved:
                es_best = float(mean_loss)
                es_bad_epochs = 0
            else:
                es_bad_epochs += 1
                if es_bad_epochs >= es_patience:
                    print(
                        f"[GSPS][prior][EarlyStop] epoch={epoch + 1} "
                        f"best={float(es_best):.6f} current={float(mean_loss):.6f}"
                    )
                    break


def _write_metrics_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        row = {k: float(metrics[k]) for k in header}
        w.writerow(row)


def _maybe_split_parts(n_nr: int, parts_cfg: object) -> List[List[int]]:
    if isinstance(parts_cfg, list) and parts_cfg:
        out: List[List[int]] = []
        for part in parts_cfg:
            if not isinstance(part, list):
                continue
            clean = [int(i) for i in part if 0 <= int(i) < n_nr]
            if clean:
                out.append(clean)
        if out:
            return out
    mid = n_nr // 2
    return [list(range(0, mid)), list(range(mid, n_nr))] if n_nr > 1 else [[0]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/eval GSPS on shared diffusion_hands preprocessing/metrics pipeline.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("GSPS config must be a YAML mapping.")

    dataset = str(cfg.get("dataset", "assembly")).strip().lower()
    data_dir = str(cfg.get("data_dir", "")).strip()
    action_filter = str(cfg.get("action_filter", ""))
    seed = int(cfg.get("seed", 0))
    gpu_index = int(cfg.get("gpu_index", 0))
    splineeqnet_root = Path(str(cfg.get("splineeqnet_root", _ROOT / "vendor" / "splineeqnet")))

    pp = _as_dict(cfg.get("preprocessing"))
    t_his = int(pp.get("input_n", 70))
    t_pred = int(pp.get("output_n", 30))
    stride = int(pp.get("stride", 5))
    time_interp = pp.get("time_interp", None)
    window_norm = pp.get("window_norm", None)
    eval_batch_mult = int(pp.get("eval_batch_mult", 1))

    mcfg = _as_dict(cfg.get("model"))
    n_pre = int(mcfg.get("n_pre", 10))
    nk = int(mcfg.get("nk", 10))
    nz = int(mcfg.get("nz", 64))
    hidden_dim = int(mcfg.get("hidden_dim", 256))
    num_stage = int(mcfg.get("num_stage", 4))
    num_flow_layer = int(mcfg.get("num_flow_layer", 3))
    parts_cfg = mcfg.get("parts")
    lambdas = [float(v) for v in mcfg.get("lambdas", [100.0, 500.0, 8.0, 25.0, 2.0, 1.0, 0.01, 0.01, 0.0])]
    alphas = [float(v) for v in mcfg.get("alphas", [100.0, 300.0])]

    tcfg = _as_dict(cfg.get("train"))
    epochs = int(tcfg.get("epochs", 100))
    batch_size = int(tcfg.get("batch_size", 64))
    lr = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    skip_if_exists = bool(tcfg.get("skip_if_exists", True))
    objective = str(tcfg.get("objective", "composite")).strip().lower()
    n_modality = int(tcfg.get("n_modality", 10))
    train_es_cfg = _as_dict(tcfg.get("early_stopping"))
    train_es_enabled = bool(tcfg.get("early_stopping_enabled", train_es_cfg.get("enabled", False)))
    train_es_patience = max(1, int(tcfg.get("early_stopping_patience", train_es_cfg.get("patience", 20))))
    train_es_min_delta = float(tcfg.get("early_stopping_min_delta", train_es_cfg.get("min_delta", 1e-4)))
    train_es_warmup = max(0, int(tcfg.get("early_stopping_warmup", train_es_cfg.get("warmup", 0))))
    train_es_monitor = str(tcfg.get("early_stopping_monitor", train_es_cfg.get("monitor", "train_loss"))).strip().lower()

    pcfg = _as_dict(cfg.get("prior"))
    prior_enabled = bool(pcfg.get("enabled", True))
    prior_epochs = int(pcfg.get("epochs", 25))
    prior_lr = float(pcfg.get("lr", 1e-2))
    prior_weight_decay = float(pcfg.get("weight_decay", 1e-3))
    prior_skip_if_exists = bool(pcfg.get("skip_if_exists", True))
    prior_es_cfg = _as_dict(pcfg.get("early_stopping"))
    prior_es_enabled = bool(pcfg.get("early_stopping_enabled", prior_es_cfg.get("enabled", False)))
    prior_es_patience = max(1, int(pcfg.get("early_stopping_patience", prior_es_cfg.get("patience", 20))))
    prior_es_min_delta = float(pcfg.get("early_stopping_min_delta", prior_es_cfg.get("min_delta", 1e-4)))
    prior_es_warmup = max(0, int(pcfg.get("early_stopping_warmup", prior_es_cfg.get("warmup", 0))))

    ecfg = _as_dict(cfg.get("eval"))
    eval_batch_size = int(ecfg.get("batch_size", max(1, batch_size * eval_batch_mult)))
    threshold = float(ecfg.get("multimodal_threshold", 0.5))

    rcfg = _as_dict(cfg.get("runtime"))
    output_dir = Path(str(rcfg.get("output_dir", _THIS_DIR / "out"))).resolve()
    metrics_csv = Path(str(rcfg.get("metrics_csv", output_dir / "eval_stats.csv"))).resolve()

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    action_tag = action_filter if action_filter else "all"
    common_tag = (
        f"gsps_{dataset}_{action_tag}_in{t_his}_out{t_pred}_"
        f"npre{n_pre}_hd{hidden_dim}_nk{nk}_nz{nz}_bs{batch_size}_ep{epochs}_obj{objective}"
    )
    gen_ckpt_path = ckpt_dir / f"{common_tag}.pt"
    prior_ckpt_path = ckpt_dir / f"{common_tag}_prior_ep{prior_epochs}_flow{num_flow_layer}.pt"

    _set_seed(seed)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        device = torch.device("cuda", index=gpu_index)

    config_mod, data_mod = _import_splineeqnet_modules(splineeqnet_root)
    DatasetCfg = config_mod.DatasetCfg
    metadata = data_mod.get_dataset_metadata(dataset)

    data_dir_eff = data_dir or str(metadata.get("default_dir", ""))
    action_filter_eff = action_filter if action_filter is not None else str(metadata.get("default_action_filter", ""))
    wrist_indices = tuple(int(i) for i in metadata.get("default_wrist_indices", (5, 26)))

    ds_cfg = DatasetCfg(
        data_dir=data_dir_eff,
        action_filter=action_filter_eff,
        input_n=t_his,
        output_n=t_pred,
        stride=stride,
        time_interp=time_interp,
        window_norm=window_norm,
        batch_size=batch_size,
        eval_batch_mult=eval_batch_mult,
        seed=seed,
        wrist_indices=wrist_indices,
        dataset=dataset,
        node_count=int(metadata.get("node_count", 21)),
        edge_index=tuple(metadata.get("edge_index", ())),
        adjacency=tuple(metadata.get("adjacency", ())),
    )

    train_dataset, val_dataset, test_dataset = data_mod.build_datasets(ds_cfg)
    train_loader, val_loader, test_loader = data_mod.make_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        seed=seed,
        eval_batch_mult=max(1, int(math.ceil(eval_batch_size / max(1, batch_size)))),
    )

    n_nodes_total = int(getattr(train_dataset, "node_count", metadata.get("node_count", 21)))
    n_nodes_nr = int(n_nodes_total - 1)
    if n_nodes_nr <= 0:
        raise RuntimeError(f"GSPS expects at least 2 joints (with root), got node_count={n_nodes_total}")

    parts = _maybe_split_parts(n_nodes_nr, parts_cfg)
    nf_specs = {
        "hidden_dim": hidden_dim,
        "num_stage": num_stage,
        "parts": parts,
        "nz": nz,
        "num_flow_layer": num_flow_layer,
    }
    gcfg = SimpleNamespace(n_pre=n_pre, nf_specs=nf_specs)
    model, pose_prior = get_model(gcfg, traj_dim=n_nodes_nr, model_type="h36m")
    model = model.to(device=device, dtype=torch.float32)
    pose_prior = pose_prior.to(device=device, dtype=torch.float32)

    # Build parent trees from shared hand links (local indexing) with root 0 for compatibility.
    hand_groups = metadata.get("hand_groups", ())
    if not hand_groups:
        raise RuntimeError(f"Missing hand_groups metadata for dataset='{dataset}'")
    links = tuple(hand_groups[0].get("links", ()))
    parents_full = _build_parents_from_links(n_nodes_total, links, root=0)
    parents_nr = [max(-1, p - 1) for p in parents_full[1:]]

    # Prior stage.
    if prior_enabled:
        if prior_ckpt_path.exists() and prior_skip_if_exists:
            pose_prior.load_state_dict(torch.load(prior_ckpt_path, map_location=device))
            print(f"[GSPS] using existing prior checkpoint: {prior_ckpt_path}")
        else:
            print(
                f"[GSPS] prior training dataset={dataset} action_filter='{action_filter_eff}' "
                f"epochs={prior_epochs} lr={prior_lr} wd={prior_weight_decay}"
            )
            _train_prior_stage(
                pose_prior=pose_prior,
                train_loader=train_loader,
                epochs=prior_epochs,
                lr=prior_lr,
                weight_decay=prior_weight_decay,
                parents_full=parents_full,
                seed=seed,
                device=device,
                early_stopping_enabled=prior_es_enabled,
                early_stopping_patience=prior_es_patience,
                early_stopping_min_delta=prior_es_min_delta,
                early_stopping_warmup=prior_es_warmup,
            )
            torch.save(pose_prior.state_dict(), prior_ckpt_path)
            print(f"[GSPS] saved prior checkpoint: {prior_ckpt_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dct_m, idct_m = get_dct_matrix(t_his + t_pred)
    dct_m = dct_m.to(device=device, dtype=torch.float32)
    idct_m = idct_m.to(device=device, dtype=torch.float32)

    if gen_ckpt_path.exists() and skip_if_exists:
        model.load_state_dict(torch.load(gen_ckpt_path, map_location=device))
        print(f"[GSPS] using existing generator checkpoint: {gen_ckpt_path}")
    else:
        print(
            f"[GSPS] training dataset={dataset} action_filter='{action_filter_eff}' "
            f"epochs={epochs} batch_size={batch_size} n_pre={n_pre} nk={nk} objective={objective}"
        )
        prior_dist = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        es_best = None
        es_bad_epochs = 0

        for epoch in range(epochs):
            model.train()
            pose_prior.eval()
            log_sum: Dict[str, float] = {}
            n_steps = 0
            for bidx, batch in enumerate(train_loader):
                inp_dct, gt_future, gt_total, _start_pose, _norm = _prepare_batch(
                    batch, t_his=t_his, t_pred=t_pred, n_pre=n_pre, dct_m=dct_m, device=device
                )

                if objective == "best_of_k":
                    pred = _sample_predictions(
                        model,
                        inp_dct=inp_dct,
                        idct_m=idct_m,
                        t_his=t_his,
                        t_pred=t_pred,
                        nk=nk,
                        nz=nz,
                        n_parts=len(parts),
                        seed=seed + epoch * 100003 + n_steps,
                        cartesian=False,
                        return_total=False,
                    )
                    loss = (pred - gt_future.unsqueeze(1)).pow(2).mean(dim=(2, 3)).min(dim=1).values.mean()
                    logs = {"loss": float(loss.item())}
                else:
                    pred_total = _sample_predictions(
                        model,
                        inp_dct=inp_dct,
                        idct_m=idct_m,
                        t_his=t_his,
                        t_pred=t_pred,
                        nk=nk,
                        nz=nz,
                        n_parts=len(parts),
                        seed=seed + epoch * 100003 + n_steps,
                        cartesian=True,
                        return_total=True,
                    )
                    gt_multi = _build_multimodal_candidates(gt_future, gt_total, t_his=t_his, n_modality=n_modality)
                    loss, logs = _composite_loss(
                        pred_total=pred_total,
                        gt_total=gt_total,
                        gt_future=gt_future,
                        gt_multi=gt_multi,
                        pose_prior=pose_prior,
                        prior_dist=prior_dist,
                        parents_full=parents_full,
                        parents_nr=parents_nr,
                        nk=nk,
                        n_parts=len(parts),
                        parts=parts,
                        alphas=alphas,
                        lambdas=lambdas,
                        t_his=t_his,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                optimizer.step()

                for k, v in logs.items():
                    log_sum[k] = log_sum.get(k, 0.0) + float(v)
                n_steps += 1

            if n_steps > 0:
                mean = {k: (v / n_steps) for k, v in log_sum.items()}
            else:
                mean = {"loss": float("nan")}
            print(f"[GSPS][epoch {epoch + 1}/{epochs}] " + " ".join(f"{k}={v:.6f}" for k, v in sorted(mean.items())))
            if train_es_enabled and (epoch + 1) > train_es_warmup:
                if train_es_monitor in {"loss", "train_loss"}:
                    monitored = float(mean.get("loss", float("nan")))
                else:
                    monitored = float(mean.get("loss", float("nan")))
                if np.isfinite(monitored):
                    improved = es_best is None or monitored < (float(es_best) - train_es_min_delta)
                    if improved:
                        es_best = monitored
                        es_bad_epochs = 0
                    else:
                        es_bad_epochs += 1
                        if es_bad_epochs >= train_es_patience:
                            print(
                                f"[GSPS][EarlyStop] epoch={epoch + 1} "
                                f"best={float(es_best):.6f} current={monitored:.6f}"
                            )
                            break

        torch.save(model.state_dict(), gen_ckpt_path)
        print(f"[GSPS] saved generator checkpoint: {gen_ckpt_path}")

    # Evaluation on test split with common canonical metrics.
    model.eval()
    sum_apd = 0.0
    sum_ade = 0.0
    sum_fde = 0.0
    sum_mmade = 0.0
    sum_mmfde = 0.0
    sum_cmd = 0.0
    sum_fid = 0.0
    total_samples = 0
    humanmac_pred_batches = []
    humanmac_tgt_batches = []
    humanmac_context_batches = []
    saved_obs = []
    saved_tgt = []
    saved_pred = []
    saved_pred_all = []

    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            inp_dct, gt_future_nr_flat, _gt_total_flat, conditioning_context_nr, norm_factor = _prepare_batch(
                batch, t_his=t_his, t_pred=t_pred, n_pre=n_pre, dct_m=dct_m, device=device
            )
            bsz = int(inp_dct.shape[0])
            pred_nr_flat = _sample_predictions(
                model,
                inp_dct=inp_dct,
                idct_m=idct_m,
                t_his=t_his,
                t_pred=t_pred,
                nk=nk,
                nz=nz,
                n_parts=len(parts),
                seed=seed + 10000019 + bidx,
                cartesian=False,
                return_total=False,
            )

            n_nr = n_nodes_nr
            pred_nr = pred_nr_flat.reshape(bsz, nk, t_pred, n_nr, 3)
            gt_nr = gt_future_nr_flat.reshape(bsz, t_pred, n_nr, 3)

            zpred = torch.zeros(bsz, nk, t_pred, 1, 3, device=device, dtype=pred_nr.dtype)
            zgt = torch.zeros(bsz, t_pred, 1, 3, device=device, dtype=gt_nr.dtype)
            zcontext = torch.zeros(bsz, t_his, 1, 3, device=device, dtype=conditioning_context_nr.dtype)
            pred_full = torch.cat([zpred, pred_nr], dim=3)
            gt_full = torch.cat([zgt, gt_nr], dim=2)
            conditioning_context_full = torch.cat([zcontext, conditioning_context_nr], dim=2)

            total_samples += bsz
            humanmac_pred_batches.append(pred_full.permute(1, 0, 2, 3, 4).detach().cpu())
            humanmac_tgt_batches.append(gt_full.detach().cpu())
            humanmac_context_batches.append(conditioning_context_full.detach().cpu())
            pred_first_all = pred_full[0]
            gt_first = gt_full[0]
            obs_first = conditioning_context_full[0]
            best_idx = torch.norm(pred_first_all - gt_first.unsqueeze(0), dim=-1).mean(dim=(1, 2)).argmin()
            saved_obs.append(obs_first.detach().cpu())
            saved_tgt.append(gt_first.detach().cpu())
            saved_pred.append(pred_first_all[best_idx].detach().cpu())
            saved_pred_all.append(pred_first_all.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("GSPS eval produced zero test samples.")

    hm = humanmac_metrics_prefixed(
        pred_candidates=torch.cat(humanmac_pred_batches, dim=1),
        gt_future=torch.cat(humanmac_tgt_batches, dim=0),
        conditioning_context=torch.cat(humanmac_context_batches, dim=0),
        threshold=threshold,
    )

    metrics = {
        "APD": float(hm["humanmac_apd"]),
        "ADE": float(hm["humanmac_ade"]),
        "FDE": float(hm["humanmac_fde"]),
        "MMADE": float(hm["humanmac_mmade"]),
        "MMFDE": float(hm["humanmac_mmfde"]),
        "CMD": float(hm["humanmac_cmd"]),
        "FID": float(hm["humanmac_fid"]),
    }
    eval_samples_path = rcfg.get("eval_samples_path")
    if eval_samples_path and saved_obs:
        save_eval_samples_npz(
            Path(str(eval_samples_path)),
            obs=torch.stack(saved_obs, dim=0),
            target=torch.stack(saved_tgt, dim=0),
            pred=torch.stack(saved_pred, dim=0),
            pred_all=torch.stack(saved_pred_all, dim=0),
            metadata={
                "model": "gsps",
                "dataset": dataset,
                "action_filter": action_filter_eff,
                "num_candidates": int(nk),
            },
        )

    _write_metrics_csv(metrics_csv, metrics)
    print("[GSPS] eval metrics:", {k: round(v, 6) for k, v in metrics.items()})
    print(f"[GSPS] metrics csv: {metrics_csv}")


if __name__ == "__main__":
    main()
