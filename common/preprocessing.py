from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np

from common.dataset_graphs import ASSEMBLY_HAND_GROUPS
def default_action_filter(dataset: str, action_filter: str) -> str:
    if action_filter:
        return action_filter
    return "pick_up_screwd" if dataset == "assembly" else ""


def split_train_val_test(data_dir: str, action_filter: str, seed: int) -> Tuple[List[str], List[str], List[str]]:
    pattern = os.path.join(data_dir, "*.npy")
    if action_filter:
        files = [path for path in sorted(glob.glob(pattern)) if action_filter in os.path.basename(path)]
    else:
        files = sorted(glob.glob(pattern))

    if not files:
        raise RuntimeError("No files found for dataset configuration.")
    if len(files) < 3:
        raise RuntimeError("At least three sequence files are required to create train/val/test splits.")

    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(files))
    rng.shuffle(idx)
    files = [os.path.abspath(files[i]) for i in idx]

    n_total = len(files)
    n_test = max(1, int(round(0.2 * n_total)))
    n_val = max(1, int(round(0.1 * n_total)))
    n_train = n_total - n_test - n_val
    if n_train < 1:
        deficit = 1 - n_train
        reducible_val = max(0, n_val - 1)
        take_val = min(deficit, reducible_val)
        n_val -= take_val
        deficit -= take_val
        if deficit > 0:
            reducible_test = max(0, n_test - 1)
            take_test = min(deficit, reducible_test)
            n_test -= take_test
        n_train = n_total - n_test - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise RuntimeError("Unable to allocate train/val/test splits with at least one file each.")

    test_start = n_total - n_test
    val_start = test_start - n_val
    return files[:val_start], files[val_start:test_start], files[test_start:]


def remove_outliers_timewise(
    coords: np.ndarray,
    k: float = 20.0,
    max_drop_frac: float = 0.3,
    max_consecutive: int = 10,
) -> Optional[np.ndarray]:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        return coords
    num_frames = coords.shape[0]
    if num_frames <= 2:
        return coords

    med = np.median(coords, axis=0, keepdims=True)
    mad = np.median(np.abs(coords - med), axis=0, keepdims=True)
    threshold = k * (mad + 1e-8)
    is_outlier_frame = np.any(np.abs(coords - med) > threshold, axis=(1, 2))

    drop_count = int(is_outlier_frame.sum())
    if drop_count / float(num_frames) > max_drop_frac:
        return None
    if drop_count > 0:
        arr = is_outlier_frame.astype(np.int32)
        padded = np.r_[0, arr, 0]
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if starts.size > 0 and int(np.max(ends - starts)) > max_consecutive:
            return None

    kept = coords[~is_outlier_frame]
    if kept.shape[0] < 3:
        return None
    return kept


def interpolate_sequence(seq: np.ndarray, time_interp: Optional[int], interpolation: str = "linear") -> np.ndarray:
    if time_interp is None or time_interp <= 0 or time_interp == seq.shape[0]:
        return seq
    t_orig = np.linspace(0.0, 1.0, seq.shape[0])
    t_new = np.linspace(0.0, 1.0, int(time_interp))
    out = np.zeros((int(time_interp), seq.shape[1], seq.shape[2]), dtype=np.float64)
    for joint_idx in range(seq.shape[1]):
        for axis_idx in range(seq.shape[2]):
            out[:, joint_idx, axis_idx] = np.interp(t_new, t_orig, seq[:, joint_idx, axis_idx])
    return out


def align_and_scale_hand_frame0_wrist_middle(
    hand: np.ndarray,
    wrist_idx: int,
    middle_idx: int = 9,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    direction = hand[0, middle_idx, :3] - hand[0, wrist_idx, :3]
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < eps:
        return hand.astype(np.float32), 1.0
    return (hand / length).astype(np.float32), length


def reorder_wrist_to_first(hand: np.ndarray, wrist_idx: int) -> np.ndarray:
    if wrist_idx == 0:
        return hand
    order = [wrist_idx] + [idx for idx in range(hand.shape[1]) if idx != wrist_idx]
    return hand[:, order, :]


def select_most_active_hand(
    file_path: str,
    time_interp: Optional[int] = None,
    window_norm: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, float]]:
    data = np.load(file_path)
    data = data.transpose(0, 1, 3, 2).reshape((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))
    data = np.transpose(data, (1, 2, 0))

    centered = data.astype(np.float64, copy=True)
    for group in ASSEMBLY_HAND_GROUPS:
        nodes = [idx for idx in group["nodes"] if idx < centered.shape[1]]
        if not nodes:
            continue
        wrist_idx = int(group["wrist_index"])
        if 0 <= wrist_idx < centered.shape[1]:
            origin = centered[:, wrist_idx, :3]
            centered[:, nodes, :3] -= origin[:, None, :]

    centered = remove_outliers_timewise(centered)
    if centered is None:
        return None

    centered = interpolate_sequence(centered, time_interp=time_interp, interpolation="linear")

    candidates: List[Tuple[np.ndarray, float, float, int]] = []
    for group in ASSEMBLY_HAND_GROUPS:
        nodes = [idx for idx in group["nodes"] if idx < centered.shape[1]]
        if not nodes:
            continue
        wrist_global = int(group["wrist_index"])
        if wrist_global not in nodes:
            continue

        local_wrist = nodes.index(wrist_global)
        hand = centered[:, nodes, :3].astype(np.float32, copy=True)
        hand, scale = align_and_scale_hand_frame0_wrist_middle(hand, wrist_idx=local_wrist, middle_idx=9)

        time_used = hand.shape[0]
        time_window = time_used if window_norm is None or window_norm <= 0 else min(time_used, int(window_norm))
        if time_window < 2:
            continue
        diff = hand[1:time_window] - hand[: time_window - 1]
        motion_energy = float(np.mean(np.linalg.norm(diff, axis=-1)))
        candidates.append((hand, float(scale), motion_energy, local_wrist))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2], reverse=True)
    best_hand, scale, _, local_wrist = candidates[0]
    best_hand = reorder_wrist_to_first(best_hand, local_wrist)
    return best_hand.astype(np.float32), float(scale)
