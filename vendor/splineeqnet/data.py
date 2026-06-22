import glob
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DatasetCfg
from datasets import AssemblyDataset, BigHandsDataset, FPHADataset, h2oDataset
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.dataset_graphs import (
    ASSEMBLY_HAND_GROUPS,
    BIGHAND_HAND_GROUPS,
    FPHA_HAND_GROUPS,
    H2O_HAND_GROUPS,
    resolve_local_hand_graph_metadata,
)


DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "assembly": {
        "dataset_cls": AssemblyDataset,
        "default_dir": "/mnt/turing-datasets/AssemblyHands/assembly101-download-scripts/data_our/",
        "default_action_filter": "pick_up_screwd",
        "hand_groups": ASSEMBLY_HAND_GROUPS,
        "default_wrist_indices": (5, 26),
        "node_count": 21,
    },
    "h2o": {
        "dataset_cls": h2oDataset,
        "default_dir": "/mnt/turing-datasets/h2o/",
        "default_action_filter": "",
        "hand_groups": H2O_HAND_GROUPS,
        "default_wrist_indices": (0, 21),
        "node_count": 21,
    },
    "bighands": {
        "dataset_cls": BigHandsDataset,
        "default_dir": "/mnt/turing-datasets/BigHands/BigHand2.2M/data/",
        "default_action_filter": "",
        "hand_groups": BIGHAND_HAND_GROUPS,
        "default_wrist_indices": (0,),
        "node_count": 21,
    },
    "fpha": {
        "dataset_cls": FPHADataset,
        "default_dir": "/mnt/turing-datasets/FPHA/data/",
        "default_action_filter": "",
        "hand_groups": FPHA_HAND_GROUPS,
        "default_wrist_indices": (0,),
        "node_count": 21,
    },
}


def get_dataset_metadata(name: str) -> dict:
    key = name.lower()
    if key not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: {', '.join(DATASET_METADATA)}")
    return DATASET_METADATA[key]


def resolve_card_hand_graph_metadata(
    dataset: str,
    wrist_indices: Tuple[int, ...] = tuple(),
) -> Dict[str, Any]:
    return dict(resolve_local_hand_graph_metadata(dataset, wrist_indices))


# ----------------------------
# Outlier removal (unchanged)
# ----------------------------

def _remove_outliers_timewise(
    coords: np.ndarray,
    k: float = 20.0,
    max_drop_frac: float = 0.3,
    max_consecutive: int = 10,
) -> Optional[np.ndarray]:
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


# ----------------------------
# Canonical alignment: frame 0 wrist->middle to (1,0,0) + scale by that bone length
# ----------------------------

def _skel_defaults_for_middle(dataset: str) -> Tuple[int, int]:
    """
    Returns (wrist_local, middle_mcp_local) for a 21-joint hand in the common ordering.

    IMPORTANT: This assumes a widely-used 21-joint layout:
      wrist=0, thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    and uses middle MCP = 9.

    Your current hand_groups use different wrist indices at the *global* level for Assembly/H2O.
    We convert to local indexing per group; this function only returns the local middle MCP index.
    """
    return (0, 9)


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Rodrigues rotation mapping unit vector a to unit vector b.
    Returns 3x3 rotation matrix R such that a @ R == b (row-vector convention).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3, dtype=np.float32)
    a = a / na
    b = b / nb

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))

    if s < eps:
        # a parallel or anti-parallel to b
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        # 180° rotation: pick any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, a))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - float(np.dot(axis, a)) * a
        axis = axis / np.maximum(np.linalg.norm(axis), eps)
        # Rodrigues with angle pi: R = I + 2[K]^2 (since sin(pi)=0, 1-cos(pi)=2)
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        R = np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
        return R.astype(np.float32)

    axis = v / s
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)
    return R.astype(np.float32)


def _align_and_scale_hand_frame0_wrist_middle(
    hand: np.ndarray,
    wrist_l: int,
    middle_l: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Assumes hand is already wrist-centered (wrist at origin) for all frames.

    Computes at frame 0:
      u = hand[0, middle_l] - hand[0, wrist_l] (but wrist_l should be 0 after centering)
    Scales entire hand by 1/||u|| so ||u|| becomes 1.
    Rotates entire hand so u becomes exactly (1,0,0).

    Returns (hand_aligned, scale_used), where scale_used is the original bone length ||u||.
    """
    u = hand[0, middle_l, :3] - hand[0, wrist_l, :3]
    L = float(np.linalg.norm(u))
    if not np.isfinite(L) or L < eps:
        # Cannot define scale/rotation; return unchanged
        return hand.astype(np.float32), 1.0

    # scale so that bone length is 1
    hand = (hand / L).astype(np.float32)

    u_scaled = hand[0, middle_l, :3] - hand[0, wrist_l, :3]
    a = u_scaled
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # R = _rotation_from_a_to_b(a, b, eps=eps)
    # hand = (hand @ R).astype(np.float32)
    return hand, L


# ----------------------------
# Loading + preprocessing (split + canonical alignment)
# ----------------------------

def calculate_embedding_wrist_centered_split_canonical(
    file: str,
    *,
    dataset: str,
    hand_groups: Tuple[dict, ...],
    time_interp: Optional[int] = None,
    window: Optional[int] = None,
    interpolation: str = "linear",
    split_hands: bool = True,
    eps: float = 1e-8,
) -> Optional[List[Tuple[np.ndarray, float]]]:
    """
    Returns a list with a SINGLE (coords_hand, norm_factor_hand):
    the hand (among the groups) with larger motion energy.
    """
    d = np.load(file)
    dataset_lower = dataset.lower()
    if dataset_lower in {"assembly"}:
        d = d.transpose(0, 1, 3, 2).reshape((d.shape[0], d.shape[1], d.shape[2] * d.shape[3]))
        d = np.transpose(d, (1, 2, 0))
    elif dataset_lower == "h2o":
        d = d.transpose(1, 3, 2, 0).reshape(d.shape[1], d.shape[2] * d.shape[3], d.shape[0])
    # fpha assumed already (T, N, C)

    T, N, C = d.shape
    if C != 3:
        raise ValueError(f"Expected 3 coordinates, got {C} for {file}")

    # wrist-centering (per group)
    centered = d.astype(np.float64, copy=True)
    for group in hand_groups:
        nodes = [idx for idx in group["nodes"] if idx < N]
        if not nodes:
            continue
        wrist_idx = int(group["wrist_index"])
        if 0 <= wrist_idx < N:
            origin = centered[:, wrist_idx, :3]
            centered[:, nodes, :3] -= origin[:, None, :]

    centered_filtered = _remove_outliers_timewise(
        centered, k=20.0, max_drop_frac=0.3, max_consecutive=10
    )
    if centered_filtered is None:
        return None

    # time interpolation
    if time_interp is not None and time_interp > 0 and time_interp != centered_filtered.shape[0]:
        T0 = centered_filtered.shape[0]
        t_orig = np.linspace(0.0, 1.0, T0)
        t_new = np.linspace(0.0, 1.0, int(time_interp))
        out = np.zeros((int(time_interp), N, 3), dtype=np.float64)
        for i in range(N):
            for k in range(3):
                if interpolation == "linear":
                    out[:, i, k] = np.interp(t_new, t_orig, centered_filtered[:, i, k])
                else:
                    try:
                        from scipy.interpolate import CubicSpline  # type: ignore
                        cs = CubicSpline(t_orig, centered_filtered[:, i, k])
                        out[:, i, k] = cs(t_new)
                    except Exception:
                        out[:, i, k] = np.interp(t_new, t_orig, centered_filtered[:, i, k])
        centered_filtered = out

    # ---- per-hand processing + motion energy ----
    hand_candidates: List[Tuple[np.ndarray, float, float]] = []
    # tuple = (hand_coords, norm_factor, motion_energy)

    for group in hand_groups:
        nodes_global = [idx for idx in group["nodes"] if idx < N]
        if not nodes_global:
            continue

        g2l = {g: li for li, g in enumerate(nodes_global)}
        wrist_g = int(group["wrist_index"])
        if wrist_g not in g2l:
            continue
        wrist_l = int(g2l[wrist_g])

        hand = centered_filtered[:, nodes_global, :3].astype(np.float32, copy=True)

        # canonical frame: wrist -> middle finger at frame 0
        _, middle_l = _skel_defaults_for_middle(dataset)
        if 0 <= middle_l < hand.shape[1]:
            hand, L = _align_and_scale_hand_frame0_wrist_middle(
                hand, wrist_l=wrist_l, middle_l=middle_l, eps=eps
            )
        else:
            L = 1.0

        # motion energy over window (or full sequence)
        T_use = hand.shape[0]
        T_win = T_use if window is None or window <= 0 else min(T_use, int(window))
        if T_win < 2:
            continue

        diff = hand[1:T_win] - hand[: T_win - 1]
        motion_energy = float(
            np.mean(np.linalg.norm(diff, axis=-1))
        )

        hand_candidates.append((hand.astype(np.float32), float(L), motion_energy))

    if not hand_candidates:
        return None

    # ---- select the most active hand ----
    hand_candidates.sort(key=lambda x: x[2], reverse=True)
    best_hand, best_norm, _ = hand_candidates[0]

    return [(best_hand, best_norm)]


# ----------------------------
# Features (unchanged)
# ----------------------------

def compute_velocity(points: np.ndarray) -> np.ndarray:
    v = np.zeros_like(points)
    v[1:] = points[1:] - points[:-1]
    denom = np.max(np.linalg.norm(v, axis=-1))
    if denom > 0:
        v = v / denom
    return v


def calculate_features(coords: np.ndarray) -> np.ndarray:
    vel = compute_velocity(coords)
    dist = np.linalg.norm(vel, axis=-1, keepdims=True)
    angle = np.divide(vel, np.maximum(dist, 1e-8))
    feats = np.concatenate([angle, dist, coords], axis=-1)
    return feats.astype(np.float32)


def collect_sequences_from_files_wrist_centered(
    files: List[str],
    *,
    node_count: int,
    hand_groups: Tuple[dict, ...],
    dataset: str,
    time_interp: Optional[int] = None,
    window_norm: Optional[int] = None,
    split_hands: bool = True,
) -> List[Tuple[np.ndarray, float]]:
    loaded: List[Tuple[np.ndarray, float]] = []
    for f in tqdm(files):
        try:
            per_hand = calculate_embedding_wrist_centered_split_canonical(
                f,
                dataset=dataset,
                hand_groups=hand_groups,
                time_interp=time_interp,
                window=window_norm,
                interpolation="linear",
                split_hands=split_hands,
            )
            if per_hand is None:
                continue
            for coords, norm_factor in per_hand:
                loaded.append((coords, norm_factor))
        except Exception:
            pass

    seqs: List[Tuple[np.ndarray, float]] = []
    for coords, norm_factor in loaded:
        feats = calculate_features(coords)
        if feats.shape[0] >= 4:
            seqs.append((feats, norm_factor))
    return seqs


# ----------------------------
# Dataset building / loaders (unchanged except split flags passthrough)
# ----------------------------

def build_datasets(ds_cfg: DatasetCfg) -> Tuple[Dataset, Dataset, Dataset]:
    metadata = get_dataset_metadata(ds_cfg.dataset)
    dataset_cls = metadata["dataset_cls"]
    node_count_full = int(ds_cfg.node_count or metadata["node_count"])

    base_hand_groups: Tuple[dict, ...] = metadata["hand_groups"]
    if ds_cfg.wrist_indices:
        if len(ds_cfg.wrist_indices) != len(base_hand_groups):
            raise ValueError("Number of wrist indices must match number of hand groups for the selected dataset.")
        hand_groups = []
        for group, wrist_idx in zip(base_hand_groups, ds_cfg.wrist_indices):
            new_group = dict(group)
            new_group["wrist_index"] = int(wrist_idx)
            hand_groups.append(new_group)
        hand_groups = tuple(hand_groups)
    else:
        hand_groups = base_hand_groups

    rng = np.random.RandomState(ds_cfg.seed)
    if ds_cfg.subset_files is not None:
        files = list(ds_cfg.subset_files)
    else:
        pat = os.path.join(ds_cfg.data_dir, "*.npy")
        if ds_cfg.action_filter != "":
            files = [f for f in sorted(glob.glob(pat)) if ds_cfg.action_filter in os.path.basename(f)]
        else:
            files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit("No files found for dataset configuration.")

    n_total = len(files)
    if n_total < 3:
        raise SystemExit("At least three sequence files are required to create train/val/test splits.")

    idx = np.arange(len(files))
    rng.shuffle(idx)
    files = [files[i] for i in idx]

    test_fraction = 0.2
    val_fraction = 0.1
    n_test = max(1, int(round(test_fraction * n_total)))
    n_val = max(1, int(round(val_fraction * n_total)))
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
            deficit -= take_test
        n_train = n_total - n_test - n_val
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise SystemExit("Unable to allocate train/val/test splits with at least one file each.")

    test_start = n_total - n_test
    val_start = test_start - n_val
    train_files = files[:val_start]
    val_files = files[val_start:test_start]
    test_files = files[test_start:]

    input_n = int(ds_cfg.input_n)
    output_n = int(ds_cfg.output_n)
    window_norm = int(ds_cfg.window_norm) if ds_cfg.window_norm is not None else input_n

    split_hands = bool(getattr(ds_cfg, "split_hands", True))

    train_sequences = collect_sequences_from_files_wrist_centered(
        train_files,
        node_count=node_count_full,
        hand_groups=hand_groups,
        dataset=ds_cfg.dataset,
        time_interp=ds_cfg.time_interp,
        window_norm=window_norm,
        split_hands=split_hands,
    )
    val_sequences = collect_sequences_from_files_wrist_centered(
        val_files,
        node_count=node_count_full,
        hand_groups=hand_groups,
        dataset=ds_cfg.dataset,
        time_interp=ds_cfg.time_interp,
        window_norm=window_norm,
        split_hands=split_hands,
    )
    test_sequences = collect_sequences_from_files_wrist_centered(
        test_files,
        node_count=node_count_full,
        hand_groups=hand_groups,
        dataset=ds_cfg.dataset,
        time_interp=ds_cfg.time_interp,
        window_norm=window_norm,
        split_hands=split_hands,
    )

    print(f"Built datasets with {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test sequences.")
    if not train_sequences or not test_sequences or not val_sequences:
        raise SystemExit("Not enough sequences to build loaders after split.")

    dset_train = dataset_cls(train_sequences, input_n=input_n, output_n=output_n, stride=ds_cfg.stride)
    dset_val = dataset_cls(val_sequences, input_n=input_n, output_n=output_n, stride=ds_cfg.stride)
    dset_test = dataset_cls(test_sequences, input_n=input_n, output_n=output_n, stride=ds_cfg.stride)
    if len(dset_train) == 0 or len(dset_test) == 0 or len(dset_val) == 0:
        raise RuntimeError("No valid windows for building DataLoaders (train/val/test).")
    return dset_train, dset_val, dset_test


def make_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    seed: int,
    eval_batch_mult: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))
    val_generator = torch.Generator()
    val_generator.manual_seed(int(seed) + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(int(seed) + 2)
    eval_mult = max(1, int(eval_batch_mult))
    eval_batch_size = max(1, int(batch_size) * eval_mult)
    pin_memory_env = os.environ.get("DIFFHF_PIN_MEMORY", "1").strip().lower()
    pin_memory = bool(torch.cuda.is_available()) and pin_memory_env not in {"0", "false", "no", "off"}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        generator=val_generator,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        generator=test_generator,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def build_loaders(ds_cfg: DatasetCfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
    return make_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        ds_cfg.batch_size,
        ds_cfg.seed,
        ds_cfg.eval_batch_mult,
    )
