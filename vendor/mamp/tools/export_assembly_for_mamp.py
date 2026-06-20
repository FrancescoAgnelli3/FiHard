#!/usr/bin/env python3
"""Export Assembly/H2O/BigHands/FPHA windows for MAMP pretraining.

This script reuses SplineEqNet preprocessing to keep representation alignment:
- wrist-centered / canonicalized hand coordinates
- same per-file filtering and interpolation path

Output NPZ contains MAMP-compatible keys:
  x_train, y_train, x_test, y_test
with x_* shaped (N, T, V, 3).
"""

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export AssemblyHands windows for MAMP.")
    p.add_argument("--splineeqnet-root", type=str, default="/home/agnelli/projects/4D_hands_working/SplineEqNet")
    p.add_argument("--dataset", type=str, default="assembly", choices=["assembly", "h2o", "bighands", "fpha"])
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--action-filter", type=str, default="")
    p.add_argument("--input-n", type=int, default=70, help="Window length T for MAMP pretraining.")
    p.add_argument("--window-stride", type=int, default=5)
    p.add_argument("--time-interp", type=int, default=None)
    p.add_argument("--window-norm", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1, help="Used together with train split to build x_test.")
    p.add_argument("--split-hands", action="store_true", default=True)
    p.add_argument("--no-split-hands", dest="split_hands", action="store_false")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _split_files(files: List[str], train_frac: float, val_frac: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    if not files:
        return [], [], []

    rng = random.Random(int(seed))
    shuffled = list(files)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = min(n_train, n - 1) if n > 1 else 1
    n_val = min(n_val, max(0, n - n_train))

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]
    return train_files, val_files, test_files


def _make_windows(seqs: List[Tuple[np.ndarray, float]], input_n: int, stride: int) -> np.ndarray:
    windows: List[np.ndarray] = []
    for feats, _norm in seqs:
        # feats shape: (T, N, 7) from SplineEqNet; coords are last 3 dims
        coords = feats[..., 4:].astype(np.float32, copy=False)
        t = coords.shape[0]
        if t < input_n:
            continue
        for st in range(0, t - input_n + 1, stride):
            windows.append(coords[st:st + input_n])

    if not windows:
        return np.zeros((0, input_n, 21, 3), dtype=np.float32)

    return np.stack(windows, axis=0).astype(np.float32, copy=False)


def _dummy_one_hot(n: int, num_classes: int = 1) -> np.ndarray:
    y = np.zeros((n, num_classes), dtype=np.float32)
    if n > 0:
        y[:, 0] = 1.0
    return y


def main() -> None:
    args = _parse_args()

    spline_root = Path(args.splineeqnet_root).resolve()
    if not spline_root.exists():
        raise FileNotFoundError(f"SplineEqNet root not found: {spline_root}")

    sys.path.insert(0, str(spline_root))
    from data import collect_sequences_from_files_wrist_centered, get_dataset_metadata  # pylint: disable=import-error

    metadata = get_dataset_metadata(args.dataset)
    node_count = int(metadata["node_count"])
    hand_groups = metadata["hand_groups"]

    pattern = os.path.join(args.data_dir, "*.npy")
    files = sorted(glob.glob(pattern))
    if args.action_filter:
        files = [f for f in files if args.action_filter in os.path.basename(f)]
    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]

    if len(files) < 2:
        raise RuntimeError(f"Need at least 2 files, found {len(files)} in {args.data_dir}")

    train_files, val_files, test_files = _split_files(files, args.train_frac, args.val_frac, args.seed)

    # Use train for x_train, and val+test for x_test to keep MAMP's train/test split contract.
    train_seqs = collect_sequences_from_files_wrist_centered(
        train_files,
        node_count=node_count,
        hand_groups=hand_groups,
        dataset=args.dataset,
        time_interp=args.time_interp,
        window_norm=args.window_norm,
        split_hands=args.split_hands,
    )
    eval_seqs = collect_sequences_from_files_wrist_centered(
        val_files + test_files,
        node_count=node_count,
        hand_groups=hand_groups,
        dataset=args.dataset,
        time_interp=args.time_interp,
        window_norm=args.window_norm,
        split_hands=args.split_hands,
    )

    x_train = _make_windows(train_seqs, input_n=args.input_n, stride=max(1, args.window_stride))
    x_test = _make_windows(eval_seqs, input_n=args.input_n, stride=max(1, args.window_stride))

    y_train = _dummy_one_hot(len(x_train), num_classes=1)
    y_test = _dummy_one_hot(len(x_test), num_classes=1)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    print(f"Saved {out_path}")
    print(f"x_train: {x_train.shape} | x_test: {x_test.shape}")
    print(f"files: train={len(train_files)} val={len(val_files)} test={len(test_files)}")


if __name__ == "__main__":
    main()
