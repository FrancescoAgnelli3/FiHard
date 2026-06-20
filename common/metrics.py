from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: TensorLike, *, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if device is not None:
            return x.to(device=device, dtype=dtype)
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _reshape_motion_sequences(x: TensorLike, *, is_prediction: bool) -> torch.Tensor:
    t = _to_tensor(x, dtype=torch.float32, device=None).detach().cpu()
    if t.dim() == 5:
        if t.shape[-1] != 3:
            raise ValueError(f"Expected last dim=3 for 5D motion tensor, got shape={tuple(t.shape)}")
        if is_prediction:
            # [K,B,T,N,3] -> [K*B,T,N,3] (also supports [B,K,T,N,3] by flattening first two axes).
            return t.reshape(-1, t.shape[2], t.shape[3], t.shape[4])
        # [B,T,N,3]
        return t
    if t.dim() == 4:
        if t.shape[-1] == 3:
            if is_prediction:
                # [S,T,N,3]
                return t
            # [B,T,N,3]
            return t
        if t.shape[-1] % 3 != 0:
            raise ValueError(f"Expected flattened xyz dim divisible by 3, got shape={tuple(t.shape)}")
        if is_prediction:
            # [K,B,T,NC] -> [K*B,T,N,3]
            return t.reshape(t.shape[0] * t.shape[1], t.shape[2], -1, 3)
        raise ValueError(f"Unsupported 4D target tensor shape={tuple(t.shape)}")
    if t.dim() == 3:
        if t.shape[-1] % 3 != 0:
            raise ValueError(f"Expected flattened xyz dim divisible by 3, got shape={tuple(t.shape)}")
        if is_prediction:
            # [S,T,NC] -> [S,T,N,3]
            return t.reshape(t.shape[0], t.shape[1], -1, 3)
        # [B,T,NC] -> [B,T,N,3]
        return t.reshape(t.shape[0], t.shape[1], -1, 3)
    raise ValueError(f"Unsupported motion tensor rank={t.dim()}, shape={tuple(t.shape)}")


def _velocity_magnitudes_per_frame(seqs_btn3: torch.Tensor) -> torch.Tensor:
    if seqs_btn3.dim() != 4 or seqs_btn3.shape[-1] != 3:
        raise ValueError(f"Expected [S,T,N,3], got shape={tuple(seqs_btn3.shape)}")
    if seqs_btn3.shape[1] <= 1:
        return torch.zeros((seqs_btn3.shape[0], 1), dtype=seqs_btn3.dtype)
    vel = torch.linalg.norm(seqs_btn3[:, 1:] - seqs_btn3[:, :-1], dim=-1)  # [S,T-1,N]
    return vel.mean(dim=-1)  # [S,T-1]


def cumulative_motion_distribution_distance(
    pred_candidates: TensorLike,
    gt_future: TensorLike,
    *,
    num_bins: int = 200,
) -> float:
    pred_seq = _reshape_motion_sequences(pred_candidates, is_prediction=True)
    gt_seq = _reshape_motion_sequences(gt_future, is_prediction=False)
    pred_motion = _velocity_magnitudes_per_frame(pred_seq).reshape(-1).numpy()
    gt_motion = _velocity_magnitudes_per_frame(gt_seq).reshape(-1).numpy()

    if pred_motion.size == 0 or gt_motion.size == 0:
        return float("nan")

    lo = float(min(pred_motion.min(), gt_motion.min()))
    hi = float(max(pred_motion.max(), gt_motion.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float("nan")
    if hi <= lo + 1e-12:
        return 0.0

    bins = np.linspace(lo, hi, int(max(2, num_bins)) + 1, dtype=np.float64)
    p_hist, _ = np.histogram(pred_motion, bins=bins)
    g_hist, _ = np.histogram(gt_motion, bins=bins)
    p_sum = int(p_hist.sum())
    g_sum = int(g_hist.sum())
    if p_sum <= 0 or g_sum <= 0:
        return float("nan")

    p_cdf = np.cumsum(p_hist.astype(np.float64) / float(p_sum))
    g_cdf = np.cumsum(g_hist.astype(np.float64) / float(g_sum))
    bin_width = (hi - lo) / float(len(p_cdf))
    return float(np.sum(np.abs(p_cdf - g_cdf)) * bin_width)


def _trajectory_features(seqs_btn3: torch.Tensor) -> np.ndarray:
    if seqs_btn3.dim() != 4 or seqs_btn3.shape[-1] != 3:
        raise ValueError(f"Expected [S,T,N,3], got shape={tuple(seqs_btn3.shape)}")
    mean_pos = seqs_btn3.mean(dim=2)  # [S,T,3]
    mean_pos = mean_pos - mean_pos[:, :1, :]
    if seqs_btn3.shape[1] <= 1:
        mean_vel = torch.zeros((seqs_btn3.shape[0], 1), dtype=seqs_btn3.dtype)
        std_vel = torch.zeros((seqs_btn3.shape[0], 1), dtype=seqs_btn3.dtype)
    else:
        vel = seqs_btn3[:, 1:] - seqs_btn3[:, :-1]  # [S,T-1,N,3]
        vel_mag = torch.linalg.norm(vel, dim=-1)  # [S,T-1,N]
        mean_vel = vel_mag.mean(dim=-1)  # [S,T-1]
        std_vel = vel_mag.std(dim=-1, unbiased=False)  # [S,T-1]
    feat = torch.cat(
        [
            mean_pos.reshape(seqs_btn3.shape[0], -1),
            mean_vel.reshape(seqs_btn3.shape[0], -1),
            std_vel.reshape(seqs_btn3.shape[0], -1),
        ],
        dim=1,
    )
    return feat.numpy().astype(np.float64, copy=False)


def _subsample_rows(arr: np.ndarray, n_rows: int) -> np.ndarray:
    if arr.shape[0] <= n_rows:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, num=n_rows, dtype=np.int64)
    return arr[idx]


def _mean_and_cov(feats: np.ndarray, *, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    if feats.shape[0] <= 1:
        cov = np.eye(feats.shape[1], dtype=np.float64) * float(eps)
        return mu, cov
    cov = np.cov(feats, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov = (cov + cov.T) * 0.5
    cov += np.eye(cov.shape[0], dtype=np.float64) * float(eps)
    return mu, cov


def frechet_motion_distance(
    pred_candidates: TensorLike,
    gt_future: TensorLike,
    *,
    max_samples: int = 5000,
    eps: float = 1e-6,
) -> float:
    pred_seq = _reshape_motion_sequences(pred_candidates, is_prediction=True)
    gt_seq = _reshape_motion_sequences(gt_future, is_prediction=False)
    pred_feat = _trajectory_features(pred_seq)
    gt_feat = _trajectory_features(gt_seq)

    if pred_feat.shape[0] == 0 or gt_feat.shape[0] == 0:
        return float("nan")
    n = int(min(pred_feat.shape[0], gt_feat.shape[0], max(1, int(max_samples))))
    pred_feat = _subsample_rows(pred_feat, n)
    gt_feat = _subsample_rows(gt_feat, n)

    mu_p, cov_p = _mean_and_cov(pred_feat, eps=eps)
    mu_g, cov_g = _mean_and_cov(gt_feat, eps=eps)

    cov_p = np.asarray(cov_p, dtype=np.float64)
    cov_g = np.asarray(cov_g, dtype=np.float64)
    cov_p = (cov_p + cov_p.T) * 0.5
    cov_g = (cov_g + cov_g.T) * 0.5

    eigvals_p, eigvecs_p = np.linalg.eigh(cov_p)
    eigvals_p = np.clip(eigvals_p, a_min=0.0, a_max=None)
    sqrt_cov_p = (eigvecs_p * np.sqrt(eigvals_p + float(eps))) @ eigvecs_p.T
    mid = sqrt_cov_p @ cov_g @ sqrt_cov_p
    mid = (mid + mid.T) * 0.5
    eigvals_mid = np.linalg.eigvalsh(mid)
    eigvals_mid = np.clip(eigvals_mid, a_min=0.0, a_max=None)
    trace_sqrt = float(np.sum(np.sqrt(eigvals_mid + float(eps))))

    diff = mu_p - mu_g
    fid = float(diff.dot(diff) + np.trace(cov_p) + np.trace(cov_g) - 2.0 * trace_sqrt)
    return max(0.0, fid)


def distributional_motion_metrics(
    pred_candidates: TensorLike,
    gt_future: TensorLike,
) -> Dict[str, float]:
    return {
        "CMD": cumulative_motion_distribution_distance(pred_candidates, gt_future),
        "FID": frechet_motion_distance(pred_candidates, gt_future),
    }


def time_slice(array: torch.Tensor, t0: int, t: int, axis: int) -> torch.Tensor:
    if t == -1:
        return torch.index_select(
            array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32)
        )
    return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))


def apd(pred: torch.Tensor, target: torch.Tensor, *args, t0: int = 0, t: int = -1) -> torch.Tensor:
    del target, args
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1:
        return torch.zeros((batch_size,), device=pred.device)
    arr = pred.reshape(batch_size, n_samples, -1)
    dist = torch.cdist(arr, arr)
    idx = torch.triu_indices(n_samples, n_samples, offset=1, device=pred.device)
    vals = dist[:, idx[0], idx[1]]
    return vals.mean(dim=1)


def ade(pred: torch.Tensor, target: torch.Tensor, *args, t0: int = 0, t: int = -1) -> torch.Tensor:
    del args
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))
    diff = pred - target
    dist = torch.linalg.norm(diff, dim=-1).mean(dim=-1)
    return dist.min(dim=-1).values


def fde(pred: torch.Tensor, target: torch.Tensor, *args, t0: int = 0, t: int = -1) -> torch.Tensor:
    del args
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))
    diff = pred - target
    dist = torch.linalg.norm(diff, dim=-1)[:, :, -1]
    return dist.min(dim=-1).values


def mmade(
    pred: torch.Tensor,
    target: torch.Tensor,
    gt_multi: Sequence[TensorLike],
    *args,
    t0: int = 0,
    t: int = -1,
) -> torch.Tensor:
    del target, args
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size,), dtype=pred.dtype, device=pred.device)
    for i in range(batch_size):
        gt_i = _to_tensor(gt_multi[i], dtype=pred.dtype, device=pred.device)
        n_gts = gt_i.shape[0]
        if n_gts == 1:
            results[i] = float("nan")
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_i, t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)
        dist = torch.linalg.norm(p - gt, dim=-1).mean(dim=-1)
        results[i] = dist.min(dim=-1).values.mean()
    return results


def mmfde(
    pred: torch.Tensor,
    target: torch.Tensor,
    gt_multi: Sequence[TensorLike],
    *args,
    t0: int = 0,
    t: int = -1,
) -> torch.Tensor:
    del target, args
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size,), dtype=pred.dtype, device=pred.device)
    for i in range(batch_size):
        gt_i = _to_tensor(gt_multi[i], dtype=pred.dtype, device=pred.device)
        n_gts = gt_i.shape[0]
        if n_gts == 1:
            results[i] = float("nan")
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_i, t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)
        dist = torch.linalg.norm(p - gt, dim=-1)[:, :, -1]
        results[i] = dist.min(dim=-1).values.mean()
    return results


def _resolve_conditioning_context(
    conditioning_context: Optional[torch.Tensor] = None,
    *,
    start_pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Keep backward compatibility with older vendor code that still passes
    # `start_pose` while the shared metrics layer now uses a more general name.
    if conditioning_context is None:
        conditioning_context = start_pose
    if conditioning_context is None:
        raise TypeError("Expected `conditioning_context` (or legacy alias `start_pose`).")
    return conditioning_context


def humanmac_metrics(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    conditioning_context: Optional[torch.Tensor] = None,
    *,
    threshold: float = 0.5,
    start_pose: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    conditioning_context = _resolve_conditioning_context(
        conditioning_context,
        start_pose=start_pose,
    )
    if pred_candidates.numel() == 0 or gt_future.numel() == 0 or conditioning_context.numel() == 0:
        return {
            "APD": float("nan"),
            "ADE": float("nan"),
            "FDE": float("nan"),
            "MMADE": float("nan"),
            "MMFDE": float("nan"),
            "CMD": float("nan"),
            "FID": float("nan"),
        }

    pred_flat = pred_candidates.reshape(
        pred_candidates.shape[0], pred_candidates.shape[1], pred_candidates.shape[2], -1
    ).cpu()
    gt_flat = gt_future.reshape(gt_future.shape[0], gt_future.shape[1], -1).cpu()
    context_flat = conditioning_context.reshape(conditioning_context.shape[0], -1).cpu()

    pairwise = torch.cdist(context_flat, context_flat)
    apd_total = 0.0
    ade_total = 0.0
    fde_total = 0.0
    mmade_total = 0.0
    mmfde_total = 0.0
    num_samples = int(gt_flat.shape[0])

    for sample_idx in range(num_samples):
        pred_sample = pred_flat[:, sample_idx, :, :]
        gt_sample = gt_flat[sample_idx : sample_idx + 1]
        group_idx = torch.nonzero(pairwise[sample_idx] < float(threshold), as_tuple=False).view(-1)
        gt_multi = gt_flat[group_idx]

        if pred_sample.shape[0] <= 1:
            apd_val = pred_sample.new_tensor(0.0)
        else:
            apd_val = torch.pdist(pred_sample.reshape(pred_sample.shape[0], -1)).mean()

        gt_multi_gt = torch.cat([gt_multi, gt_sample], dim=0)
        dist = torch.linalg.norm(pred_sample[:, None, :, :] - gt_multi_gt[None, :, :, :], dim=3)

        mmfde_val = dist[:, :-1, -1].min(dim=0).values.mean()
        mmade_val = dist[:, :-1].mean(dim=2).min(dim=0).values.mean()
        ade_val = dist[:, -1].mean(dim=1).min(dim=0).values.mean()
        fde_val = dist[:, -1, -1].min(dim=0).values.mean()

        apd_total += float(apd_val.item())
        ade_total += float(ade_val.item())
        fde_total += float(fde_val.item())
        mmade_total += float(mmade_val.item())
        mmfde_total += float(mmfde_val.item())

    denom = max(1, num_samples)
    out = {
        "APD": apd_total / denom,
        "ADE": ade_total / denom,
        "FDE": fde_total / denom,
        "MMADE": mmade_total / denom,
        "MMFDE": mmfde_total / denom,
    }
    out.update(distributional_motion_metrics(pred_candidates, gt_future))
    return out


def humanmac_metrics_prefixed(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    conditioning_context: Optional[torch.Tensor] = None,
    *,
    threshold: float = 0.5,
    start_pose: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    m = humanmac_metrics(
        pred_candidates,
        gt_future,
        conditioning_context,
        threshold=threshold,
        start_pose=start_pose,
    )
    return {
        "humanmac_apd": m["APD"],
        "humanmac_ade": m["ADE"],
        "humanmac_fde": m["FDE"],
        "humanmac_mmade": m["MMADE"],
        "humanmac_mmfde": m["MMFDE"],
        "humanmac_cmd": m["CMD"],
        "humanmac_fid": m["FID"],
    }


def splineeqnet_diffusion_batch_eval(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    conditioning_context: Optional[torch.Tensor] = None,
    norm_factor: Optional[torch.Tensor] = None,
    *,
    threshold: float = 0.5,
    start_pose: Optional[torch.Tensor] = None,
) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
    conditioning_context = _resolve_conditioning_context(
        conditioning_context,
        start_pose=start_pose,
    )
    if norm_factor is None:
        raise TypeError("Expected `norm_factor`.")
    if pred_candidates.dim() != 4 or gt_future.dim() != 3:
        raise ValueError("Expected pred_candidates[K,B,T,NC] and gt_future[B,T,NC].")
    if pred_candidates.shape[3] % 3 != 0:
        raise ValueError("Last dim (NC) must be divisible by 3.")

    pred_bktc = pred_candidates.permute(1, 0, 2, 3).contiguous()
    b, k, t, nc = pred_bktc.shape
    n = nc // 3
    pred_bktn3 = pred_bktc.reshape(b, k, t, n, 3)
    gt_btn3 = gt_future.reshape(b, t, n, 3)

    humanmac = humanmac_metrics(pred_candidates, gt_future, conditioning_context, threshold=threshold)
    return {
        "humanmac": humanmac,
    }


def compute_all_metrics_single(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_multi: TensorLike,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = _to_tensor(pred, dtype=torch.float32, device=pred.device if isinstance(pred, torch.Tensor) else None)
    gt = _to_tensor(gt, dtype=torch.float32, device=pred.device)
    gt_multi_t = _to_tensor(gt_multi, dtype=torch.float32, device=pred.device)

    if pred.shape[0] == 1:
        diversity = pred.new_tensor(0.0)
    else:
        diversity = torch.pdist(pred.reshape(pred.shape[0], -1)).mean()

    gt_multi_gt = torch.cat([gt_multi_t, gt], dim=0)[None, ...]
    pred_b = pred[:, None, ...]
    dist = torch.linalg.norm(pred_b - gt_multi_gt, dim=3)

    mmfde = dist[:, :-1, -1].min(dim=0).values.mean()
    mmade = dist[:, :-1].mean(dim=2).min(dim=0).values.mean()
    ade_v = dist[:, -1].mean(dim=1).min(dim=0).values.mean()
    fde_v = dist[:, -1, -1].min(dim=0).values.mean()
    return diversity, ade_v, fde_v, mmade, mmfde
