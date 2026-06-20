import os
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_coords(y: np.ndarray, window: Optional[int] = None, mode: str = "global") -> Tuple[np.ndarray, float]:
    assert y.ndim == 3  # (T, N, 3)
    if mode == "past" and window is not None and window > 0:
        max_norm = np.max(np.linalg.norm(y[:window], axis=-1))
    else:
        max_norm = np.max(np.linalg.norm(y, axis=-1))
    if max_norm == 0:
        return y, 1.0
    return y / max_norm, float(max_norm)


def compute_velocity(points: np.ndarray) -> np.ndarray:
    v = np.zeros_like(points)
    v[1:] = points[1:] - points[:-1]
    denom = np.max(np.linalg.norm(points, axis=-1))
    if denom > 0:
        v = v / denom
    return v


def weighted_joint_loss(joint_weights: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, metric: str = 'mae', dims: Tuple[int, int] = (1, 2)) -> torch.Tensor:
    if metric == 'mae':
        per_joint_error = (pred - target).abs().sum(dim=-1)
    elif metric == 'mse':
        per_joint_error = (pred - target).pow(2).sum(dim=-1)
    else:
        raise ValueError("metric must be 'mae' or 'mse'")
    weighted_error = per_joint_error * joint_weights.view(1, 1, -1)
    for dim in sorted(dims, reverse=True):
        weighted_error = weighted_error.mean(dim=dim)
    weighted_error = weighted_error.sum(dim=0)
    return weighted_error


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    cov = (pred_centered * target_centered).mean(dim=1)
    pred_var = pred_centered.pow(2).mean(dim=1)
    target_var = target_centered.pow(2).mean(dim=1)
    denom = torch.sqrt(pred_var * target_var + eps)
    corr = cov / denom
    return (1.0 - corr).mean()


def compute_node_motion_weights(data_loader: DataLoader, device: torch.device, normalize: str = 'sum') -> torch.Tensor:
    total_motion = None
    total_count = 0
    for batch in data_loader:
        inputs = batch[0].to(device)
        x = inputs[:, :, :, 4:7]
        dx = x[:, 1:] - x[:, :-1]
        motion = dx.norm(dim=-1).sum(dim=1)
        if total_motion is None:
            total_motion = motion.sum(dim=0)
        else:
            total_motion += motion.sum(dim=0)
        total_count += x.size(0)
    if total_motion is None or total_count == 0:
        return torch.ones(1, device=device)
    avg_motion = total_motion / total_count
    if normalize == 'sum':
        weights = avg_motion / (avg_motion.sum() + 1e-8)
    elif normalize == 'max':
        weights = avg_motion / (avg_motion.max() + 1e-8)
    else:
        weights = avg_motion
    return weights


def _remove_outliers_timewise(coords: np.ndarray, k: float = 5.0, max_drop_frac: float = 0.3, max_consecutive: int = 10) -> Optional[np.ndarray]:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        return coords
    T = coords.shape[0]
    if T <= 2:
        return coords
    med = np.median(coords, axis=0, keepdims=True)
    mad = np.median(np.abs(coords - med), axis=0, keepdims=True)
    thr = k * (mad + 1e-8)
    dev = np.abs(coords - med)
    is_outlier_comp = dev > thr
    is_outlier_frame = np.any(is_outlier_comp, axis=(1, 2))
    drop_count = int(is_outlier_frame.sum())
    if drop_count / float(T) > max_drop_frac:
        return None
    if drop_count > 0:
        arr = is_outlier_frame.astype(np.int32)
        padded = np.r_[0, arr, 0]
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if starts.size > 0:
            max_run = int(np.max(ends - starts))
            if max_run > max_consecutive:
                return None
    kept = coords[~is_outlier_frame]
    if kept.shape[0] < 3:
        return None
    return kept


def calculate_embedding(file: str, *, time_interp: Optional[int] = None, window: Optional[int] = None,
                        interpolation: str = "linear", coord_frame: str = "midpoint") -> Optional[Tuple[np.ndarray, float]]:
    d = np.load(file)
    d = d.transpose(0, 1, 3, 2).reshape((d.shape[0], d.shape[1], d.shape[2] * d.shape[3]))
    d = np.transpose(d, (1, 2, 0))  # (t, n, 3)
    T, N, C = d.shape
    if C != 3:
        raise ValueError(f"Expected 3 coordinates, got {C} for {file}")
    if time_interp is not None and time_interp > 0 and time_interp != T:
        t_orig = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, time_interp)
        out = np.zeros((time_interp, N, 3), dtype=d.dtype)
        for i in range(N):
            for k in range(3):
                if interpolation == "linear":
                    out[:, i, k] = np.interp(t_new, t_orig, d[:, i, k])
                else:
                    try:
                        from scipy.interpolate import CubicSpline
                        cs = CubicSpline(t_orig, d[:, i, k])
                        out[:, i, k] = cs(t_new)
                    except Exception:
                        out[:, i, k] = np.interp(t_new, t_orig, d[:, i, k])
        d = out
        T = time_interp
    if coord_frame in ("per_hand", "midpoint") and N >= 42:
        left_idx = slice(0, 21)
        right_idx = slice(21, 42)
        left_center = d[:, left_idx, :3].mean(axis=1, keepdims=True)
        right_center = d[:, right_idx, :3].mean(axis=1, keepdims=True)
        if coord_frame == "per_hand":
            d[:, left_idx, :3] = d[:, left_idx, :3] - left_center
            d[:, right_idx, :3] = d[:, right_idx, :3] - right_center
        elif coord_frame == "midpoint":
            midpoint = 0.5 * (left_center + right_center)
            d[:, :, :3] = d[:, :, :3] - midpoint
    else:
        scene_center = d[:, :, :3].mean(axis=1, keepdims=True)
        d[:, :, :3] = d[:, :, :3] - scene_center
    d_filtered = _remove_outliers_timewise(d, k=20.0, max_drop_frac=0.3, max_consecutive=10)
    if d_filtered is None:
        return None
    d, norm_factor = normalize_coords(d_filtered, window=window, mode="past")
    return d.astype(np.float32), float(norm_factor)


def to_mpt_features(coords: np.ndarray) -> np.ndarray:
    T, N, _ = coords.shape
    vel = compute_velocity(coords)
    dist = np.linalg.norm(vel, axis=-1, keepdims=True)
    angle = np.divide(vel, np.maximum(dist, 1e-8))
    feats = np.concatenate([angle, dist, coords], axis=-1)
    return feats.astype(np.float32)


def collect_sequences_from_files(files: list, *, nodes: int = 42, time_interp: Optional[int] = None,
                                 window_norm: Optional[int] = None) -> List[Tuple[np.ndarray, float]]:
    loaded: List[Tuple[np.ndarray, float]] = []
    i = 0
    for f in tqdm(files):
        try:
            embedding = calculate_embedding(
                f,
                time_interp=time_interp,
                window=window_norm,
                interpolation="linear",
                coord_frame="midpoint",
            )
            if embedding is None:
                continue
            coords, norm_factor = embedding
            if coords.shape[1] != nodes:
                continue
            if coords.shape[0] < 4:
                continue
            loaded.append((coords, norm_factor))
        except Exception:
            print(f"Warning: failed to load {f}")
    seqs: List[Tuple[np.ndarray, float]] = []
    for coords, norm_factor in loaded:
        d = coords[1:] - coords[:-1]
        mean_d = np.linalg.norm(d, axis=-1).mean(axis=1)
        if mean_d.max() > 0.2:
            continue
        feats = to_mpt_features(coords)
        if feats.shape[0] >= 4:
            seqs.append((feats, norm_factor))
    return seqs


def reconstruct_sequence(pred_v: torch.Tensor, pred_angles: torch.Tensor, init_pose: torch.Tensor, node_num: int) -> torch.Tensor:
    B, T_out, N = pred_v.shape
    steps = []
    cur = init_pose
    for t in range(T_out):
        step = cur + pred_v[:, t].unsqueeze(-1) * pred_angles[:, t]
        steps.append(step)
        cur = step
    return torch.stack(steps, dim=1)


def find_semskeconvs(*roots):
    import model_4GRU  # local import to avoid circulars
    layers = []
    for root in roots:
        if root is None:
            continue
        for name, mod in root.named_modules():
            if isinstance(mod, model_4GRU.SemskeConv):
                layers.append((f"{root.__class__.__name__}.{name}", mod))
    return layers


def semskeconv_stats(layers) -> dict:
    stats = {}
    for name, mod in layers:
        try:
            W = mod.W.detach().float().cpu()
            M = mod.M.detach().float().cpu()
            A = mod.A_sem.detach().float().cpu() if hasattr(mod, 'A_sem') else None
            stats[name] = {
                'W_l2': float(W.norm(p=2)),
                'W_maxabs': float(W.abs().max()),
                'M_l2': float(M.norm(p=2)),
                'M_maxabs': float(M.abs().max()),
                'A_sem_l2': float(A.norm(p=2)) if A is not None else None,
                'A_sem_maxabs': float(A.abs().max()) if A is not None else None,
            }
        except Exception as e:
            stats[name] = {'error': str(e)}
    return stats


def print_semskeconv_stats(stats: dict, prefix: str = "[GCN]"):
    names = sorted(stats.keys())
    for n in names:
        s = stats[n]
        if 'error' in s:
            print(f"{prefix} {n}: error={s['error']}")
        else:
            print(f"{prefix} {n}: W_l2={s['W_l2']:.6e} W_max={s['W_maxabs']:.3e} | M_l2={s['M_l2']:.6e} M_max={s['M_maxabs']:.3e} | A_sem_l2={s['A_sem_l2']} A_sem_max={s['A_sem_maxabs']}")


def bone_length_loss_edges(pred: torch.Tensor, tgt: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    i = edges[:, 0]
    j = edges[:, 1]
    pred_i = pred[..., i, :]
    pred_j = pred[..., j, :]
    tgt_i = tgt[..., i, :]
    tgt_j = tgt[..., j, :]
    pred_len = torch.norm(pred_i - pred_j, dim=-1)
    tgt_len = torch.norm(tgt_i - tgt_j, dim=-1)
    return torch.mean((pred_len - tgt_len) ** 2)
