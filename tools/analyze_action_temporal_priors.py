#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.preprocessing import select_most_active_hand, split_train_val_test


ABLATION_MODEL_TO_FAMILY = {
    "FiHard_temporal_velocity_prior_q025": "velocity",
    "FiHard_temporal_acceleration_prior_q025": "acceleration",
    "FiHard_temporal_jerk_prior_q025": "jerk",
    "FiHard_temporal_velocity_acceleration_jerk_learned_prior_q025": "learned_mix",
}
FAMILY_ORDER = ("velocity", "acceleration", "jerk")


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _render_progress(current: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[no files]"
    ratio = min(max(float(current) / float(total), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * max(0, width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100.0:5.1f}%)"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _best_ablation_rows(path: Path) -> Dict[Tuple[str, str], dict]:
    if not path.exists():
        return {}

    by_key: Dict[Tuple[str, str], List[dict]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            action = str(row.get("action_filter", "")).strip()
            model = str(row.get("model", "")).strip()
            if not action or model not in ABLATION_MODEL_TO_FAMILY:
                continue
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            try:
                float(row["ADE"])
                float(row["FDE"])
                float(row["MMADE"])
                float(row["MMFDE"])
                float(row["FID"])
                float(row["APD"])
            except (KeyError, TypeError, ValueError):
                continue
            by_key.setdefault((action, model), []).append(row)

    best: Dict[Tuple[str, str], dict] = {}
    for key, rows in by_key.items():
        rows.sort(
            key=lambda r: (
                float(r["ADE"]),
                float(r["FDE"]),
                float(r["MMADE"]),
                float(r["MMFDE"]),
                float(r["FID"]),
                -float(r["APD"]),
            )
        )
        best[key] = rows[0]
    return best


def _ablation_winners(best_rows: Dict[Tuple[str, str], dict]) -> Dict[str, dict]:
    by_action: Dict[str, List[dict]] = {}
    for (action, _model), row in best_rows.items():
        by_action.setdefault(action, []).append(row)

    winners: Dict[str, dict] = {}
    for action, rows in by_action.items():
        rows.sort(
            key=lambda r: (
                float(r["ADE"]),
                float(r["FDE"]),
                float(r["MMADE"]),
                float(r["MMFDE"]),
                float(r["FID"]),
                -float(r["APD"]),
            )
        )
        winners[action] = rows[0]
    return winners


def _discover_actions(cfg: dict, ablation_winners: Dict[str, dict], actions_arg: Optional[str]) -> List[str]:
    if actions_arg:
        return sorted({part.strip() for part in actions_arg.split(",") if part.strip()})

    configured = cfg.get("action_filter", [])
    actions: List[str] = []
    if isinstance(configured, str) and configured.strip():
        actions.append(configured.strip())
    elif isinstance(configured, Sequence):
        for item in configured:
            text = str(item).strip()
            if text:
                actions.append(text)

    for action in ablation_winners:
        if action:
            actions.append(action)
    return sorted(set(actions))


def _resolve_data_dir(cfg: dict) -> Path:
    data_roots = dict(cfg.get("data_roots", {}) or {})
    data_dir = str(data_roots.get("assembly", "")).strip()
    if not data_dir:
        raise ValueError("No Assembly data root found in config under data_roots.assembly")
    out = Path(data_dir).expanduser()
    if not out.exists():
        raise FileNotFoundError(f"Assembly data directory does not exist: {out}")
    return out


def _extract_preprocessing(cfg: dict) -> dict:
    prep = dict(cfg.get("preprocessing", {}) or {})
    input_n = int(prep.get("input_n", 70))
    output_n = int(prep.get("output_n", 30))
    stride = int(prep.get("stride", 5))
    time_interp = prep.get("time_interp")
    window_norm = prep.get("window_norm")
    return {
        "input_n": input_n,
        "output_n": output_n,
        "stride": stride,
        "time_interp": None if time_interp in (None, "", "null", "None") else int(time_interp),
        "window_norm": None if window_norm in (None, "", "null", "None") else int(window_norm),
    }


def _files_for_action(data_dir: Path, action: str, split: str, seed: int) -> List[Path]:
    train_files, val_files, test_files = split_train_val_test(str(data_dir), action, seed=seed)
    if split == "train":
        raw = train_files
    elif split == "val":
        raw = val_files
    elif split == "test":
        raw = test_files
    else:
        raw = list(train_files) + list(val_files) + list(test_files)
    return [Path(p) for p in raw]


def _subsample_files(files: Sequence[Path], max_files: Optional[int], seed: int) -> List[Path]:
    items = list(files)
    if max_files is None or max_files <= 0 or len(items) <= max_files:
        return items
    rng = np.random.RandomState(int(seed))
    chosen = np.sort(rng.choice(len(items), size=int(max_files), replace=False))
    return [items[int(idx)] for idx in chosen]


def _extract_future_windows(hand_xyz: np.ndarray, input_n: int, output_n: int, stride: int) -> List[np.ndarray]:
    total = int(input_n) + int(output_n)
    if hand_xyz.shape[0] < total:
        return []
    windows: List[np.ndarray] = []
    for start in range(0, hand_xyz.shape[0] - total + 1, int(stride)):
        future = hand_xyz[start + input_n : start + total]
        if future.shape[0] == output_n:
            windows.append(future.astype(np.float32, copy=False))
    return windows


def _bootstrap_from_file_values(
    file_values: np.ndarray,
    rng: np.random.RandomState,
    num_bootstrap: int,
    alpha: float,
) -> Tuple[float, float]:
    if file_values.size == 0:
        return float("nan"), float("nan")
    if file_values.size == 1:
        value = float(file_values[0])
        return value, value
    idx = rng.randint(0, file_values.size, size=(num_bootstrap, file_values.size))
    means = file_values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _poly_fit_rss(window_xyz: np.ndarray, degree: int) -> float:
    arr = window_xyz.reshape(window_xyz.shape[0], -1).astype(np.float64, copy=False)
    time = np.linspace(-1.0, 1.0, arr.shape[0], dtype=np.float64)
    design = np.vander(time, N=int(degree) + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(design, arr, rcond=None)
    fitted = design @ coeffs
    residual = arr - fitted
    return float(np.sum(residual * residual))


def _finite_difference_energy(window_xyz: np.ndarray, order: int) -> float:
    arr = window_xyz.astype(np.float64, copy=False)
    diff = arr
    for _ in range(int(order)):
        diff = np.diff(diff, axis=0)
    if diff.size == 0:
        return float("nan")
    return float(np.mean(np.sum(diff * diff, axis=-1)))


def _curvature_score(window_xyz: np.ndarray, *, eps: float = 1e-12) -> Tuple[float, float, float]:
    velocity_energy = _finite_difference_energy(window_xyz, order=1)
    acceleration_energy = _finite_difference_energy(window_xyz, order=2)
    if not math.isfinite(velocity_energy) or not math.isfinite(acceleration_energy):
        return float("nan"), velocity_energy, acceleration_energy
    return float(acceleration_energy / max(velocity_energy, eps)), velocity_energy, acceleration_energy


def _sample_prior_scores(window_xyz: np.ndarray) -> dict:
    rss0 = max(_poly_fit_rss(window_xyz, degree=0), 1e-12)
    rss1 = max(_poly_fit_rss(window_xyz, degree=1), 1e-12)
    rss2 = max(_poly_fit_rss(window_xyz, degree=2), 1e-12)

    n_obs = int(np.prod(window_xyz.shape))
    dim = int(np.prod(window_xyz.shape[1:]))
    bic0 = n_obs * math.log(rss0 / n_obs) + (1 * dim) * math.log(n_obs)
    bic1 = n_obs * math.log(rss1 / n_obs) + (2 * dim) * math.log(n_obs)
    bic2 = n_obs * math.log(rss2 / n_obs) + (3 * dim) * math.log(n_obs)

    bic_by_family = {
        "velocity": float(bic0),
        "acceleration": float(bic1),
        "jerk": float(bic2),
    }
    best_family = min(bic_by_family, key=bic_by_family.get)
    best_bic = bic_by_family[best_family]
    second_bic = sorted(bic_by_family.values())[1]

    curvature_score, velocity_energy, acceleration_energy = _curvature_score(window_xyz)

    out = {
        "curvature_score": curvature_score,
        "velocity_energy": velocity_energy,
        "acceleration_energy": acceleration_energy,
        "rss_constant": float(rss0),
        "rss_linear": float(rss1),
        "rss_quadratic": float(rss2),
        "r2_linear_vs_constant": float(1.0 - (rss1 / rss0)),
        "r2_quadratic_vs_constant": float(1.0 - (rss2 / rss0)),
        "bic_velocity": float(bic0),
        "bic_acceleration": float(bic1),
        "bic_jerk": float(bic2),
        "bic_margin": float(second_bic - best_bic),
        "best_family": best_family,
    }
    return out


def _summarize_action(
    action: str,
    files: Sequence[Path],
    *,
    input_n: int,
    output_n: int,
    stride: int,
    time_interp: Optional[int],
    window_norm: Optional[int],
    bootstrap_samples: int,
    ci_alpha: float,
    bootstrap_seed: int,
) -> dict:
    _log(f"Starting action '{action}' on {len(files)} files")
    rows: List[dict] = []
    file_rows: List[List[dict]] = []
    accepted_files = 0
    total_files = len(files)
    progress_every = max(1, total_files // 20) if total_files > 0 else 1
    start_time = time.time()
    for file_idx, file_path in enumerate(files, start=1):
        selected = select_most_active_hand(
            str(file_path),
            time_interp=time_interp,
            window_norm=window_norm,
        )
        if selected is None:
            continue
        hand_xyz, _scale = selected
        windows = _extract_future_windows(hand_xyz, input_n=input_n, output_n=output_n, stride=stride)
        if not windows:
            continue
        accepted_files += 1
        current_file_rows: List[dict] = []
        for window in windows:
            row = _sample_prior_scores(window)
            rows.append(row)
            current_file_rows.append(row)
        if current_file_rows:
            file_rows.append(current_file_rows)
        if file_idx == total_files or file_idx % progress_every == 0:
            elapsed = time.time() - start_time
            _log(
                f"{action}: {_render_progress(file_idx, total_files)} | "
                f"accepted={accepted_files} | windows={len(rows)} | elapsed={elapsed:.1f}s"
            )

    if not rows:
        _log(f"Action '{action}' produced no usable windows")
        return {
            "action": action,
            "file_count": len(files),
            "accepted_file_count": accepted_files,
            "window_count": 0,
        }

    def _mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    family_counts = {family: 0 for family in FAMILY_ORDER}
    for row in rows:
        family_counts[row["best_family"]] += 1
    heuristic_family = max(
        FAMILY_ORDER,
        key=lambda family: (family_counts[family], -_mean(f"bic_{family}")),
    )

    rng = np.random.RandomState(int(bootstrap_seed))
    per_file_stats: Dict[str, np.ndarray] = {}
    per_file_stats["mean_curvature_score"] = np.asarray(
        [np.mean([row["curvature_score"] for row in chunk]) for chunk in file_rows],
        dtype=np.float64,
    )
    curvature_lo, curvature_hi = _bootstrap_from_file_values(
        per_file_stats["mean_curvature_score"], rng, bootstrap_samples, ci_alpha
    )

    summary = {
        "action": action,
        "file_count": len(files),
        "accepted_file_count": accepted_files,
        "window_count": len(rows),
        "mean_curvature_score": _mean("curvature_score"),
        "ci_curvature_score_lo": curvature_lo,
        "ci_curvature_score_hi": curvature_hi,
        "mean_velocity_energy": _mean("velocity_energy"),
        "mean_acceleration_energy": _mean("acceleration_energy"),
        "mean_r2_linear_vs_constant": _mean("r2_linear_vs_constant"),
        "mean_r2_quadratic_vs_constant": _mean("r2_quadratic_vs_constant"),
        "mean_bic_velocity": _mean("bic_velocity"),
        "mean_bic_acceleration": _mean("bic_acceleration"),
        "mean_bic_jerk": _mean("bic_jerk"),
        "mean_bic_margin": _mean("bic_margin"),
        "heuristic_distribution_winner": heuristic_family,
    }
    for family in FAMILY_ORDER:
        summary[f"bic_vote_share_{family}"] = float(family_counts[family] / len(rows))
    _log(
        f"Finished action '{action}' | usable_files={accepted_files}/{len(files)} | "
        f"windows={len(rows)} | mean_curvature_score={summary['mean_curvature_score']:.6g}"
    )
    return summary


def _parse_curvature_thresholds(value: str) -> Optional[Tuple[float, float]]:
    text = str(value or "").strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--curvature-thresholds must be two comma-separated values: low_to_mid,mid_to_high")
    low_mid, mid_high = float(parts[0]), float(parts[1])
    if not low_mid < mid_high:
        raise ValueError("--curvature-thresholds values must be increasing")
    return low_mid, mid_high


def _calibrate_curvature_thresholds(summary_rows: Sequence[dict]) -> Tuple[float, float]:
    scores = np.asarray(
        [
            float(row["mean_curvature_score"])
            for row in summary_rows
            if row.get("window_count", 0) and math.isfinite(float(row.get("mean_curvature_score", float("nan"))))
        ],
        dtype=np.float64,
    )
    if scores.size == 0:
        return float("nan"), float("nan")
    if scores.size == 1:
        value = float(scores[0])
        return value, value
    return float(np.quantile(scores, 1.0 / 3.0)), float(np.quantile(scores, 2.0 / 3.0))


def _classify_curvature(curvature: float, thresholds: Tuple[float, float]) -> str:
    low_mid, mid_high = thresholds
    if not math.isfinite(float(curvature)) or not math.isfinite(low_mid) or not math.isfinite(mid_high):
        return ""
    if curvature <= low_mid:
        return "velocity"
    if curvature <= mid_high:
        return "acceleration"
    return "jerk"


def _apply_curvature_classification(
    summary_rows: Sequence[dict],
    thresholds: Tuple[float, float],
    *,
    threshold_source: str,
) -> List[dict]:
    classified: List[dict] = []
    low_mid, mid_high = thresholds
    for row in summary_rows:
        out = dict(row)
        out["curvature_low_mid_threshold"] = low_mid
        out["curvature_mid_high_threshold"] = mid_high
        out["curvature_threshold_source"] = threshold_source
        out["curvature_winner"] = _classify_curvature(
            float(out.get("mean_curvature_score", float("nan"))),
            thresholds,
        )
        classified.append(out)
    return classified


def _merge_with_ablation(summary_rows: List[dict], ablation_winners: Dict[str, dict]) -> List[dict]:
    merged: List[dict] = []
    for row in summary_rows:
        action = row["action"]
        winner_row = ablation_winners.get(action)
        merged_row = dict(row)
        if winner_row is None:
            merged_row["ablation_winner_model"] = ""
            merged_row["ablation_winner_family"] = ""
            merged_row["ablation_winner_ade"] = ""
            merged_row["distribution_matches_ablation"] = ""
            merged_row["curvature_matches_ablation"] = ""
        else:
            family = ABLATION_MODEL_TO_FAMILY[str(winner_row["model"])]
            merged_row["ablation_winner_model"] = str(winner_row["model"])
            merged_row["ablation_winner_family"] = family
            merged_row["ablation_winner_ade"] = float(winner_row["ADE"])
            merged_row["distribution_matches_ablation"] = str(
                family == row.get("heuristic_distribution_winner", "")
            ).lower()
            merged_row["curvature_matches_ablation"] = str(family == row.get("curvature_winner", "")).lower()
        merged.append(merged_row)
    return merged


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "action",
        "file_count",
        "accepted_file_count",
        "window_count",
        "mean_curvature_score",
        "ci_curvature_score_lo",
        "ci_curvature_score_hi",
        "mean_velocity_energy",
        "mean_acceleration_energy",
        "curvature_low_mid_threshold",
        "curvature_mid_high_threshold",
        "curvature_threshold_source",
        "curvature_winner",
        "mean_r2_linear_vs_constant",
        "mean_r2_quadratic_vs_constant",
        "mean_bic_velocity",
        "mean_bic_acceleration",
        "mean_bic_jerk",
        "mean_bic_margin",
        "bic_vote_share_velocity",
        "bic_vote_share_acceleration",
        "bic_vote_share_jerk",
        "heuristic_distribution_winner",
        "ablation_winner_model",
        "ablation_winner_family",
        "ablation_winner_ade",
        "distribution_matches_ablation",
        "curvature_matches_ablation",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def _format_float(value: object, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def _write_markdown(
    path: Path,
    rows: Sequence[dict],
    *,
    split: str,
    input_n: int,
    output_n: int,
    stride: int,
    data_dir: Path,
    max_files_per_action: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Action-Conditioned Temporal Prior Analysis\n\n")
        f.write("This report classifies Assembly actions with one normalized curvature score: `acceleration_energy / velocity_energy`.\n\n")
        f.write("- low curvature -> `velocity` prior bias\n")
        f.write("- middle curvature -> `acceleration` prior bias\n")
        f.write("- high curvature -> `jerk` prior bias\n\n")
        f.write("By default, low/middle/high are calibrated as tertiles over the analyzed actions. Pass `--curvature-thresholds low,high` to use fixed absolute thresholds.\n\n")
        f.write(f"- data_dir: `{data_dir}`\n")
        f.write(f"- split: `{split}`\n")
        f.write(f"- obs_length: `{input_n}`\n")
        f.write(f"- pred_length: `{output_n}`\n")
        f.write(f"- stride: `{stride}`\n\n")
        f.write(f"- max_files_per_action: `{max_files_per_action}` (`0` means disabled)\n\n")
        if rows:
            first = rows[0]
            f.write(f"- low_to_mid_threshold: `{_format_float(first.get('curvature_low_mid_threshold', ''), 6)}`\n")
            f.write(f"- mid_to_high_threshold: `{_format_float(first.get('curvature_mid_high_threshold', ''), 6)}`\n")
            f.write(f"- threshold_source: `{first.get('curvature_threshold_source', '')}`\n\n")
        f.write("| action | windows | mean curvature | curvature CI | velocity energy | acceleration energy | curvature winner | ablation winner | curvature match | distribution proxy winner | distribution match |\n")
        f.write("|---|---:|---:|---|---:|---:|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                "| {action} | {window_count} | {curvature} | {curvature_ci} | {velocity_energy} | {acceleration_energy} | {curvature_winner} | {ablation_winner_family} | {curvature_match} | {distribution_winner} | {distribution_match} |\n".format(
                    action=row["action"],
                    window_count=row.get("window_count", 0),
                    curvature=_format_float(row.get("mean_curvature_score", ""), digits=6),
                    curvature_ci=f"{_format_float(row.get('ci_curvature_score_lo', ''), 6)}..{_format_float(row.get('ci_curvature_score_hi', ''), 6)}",
                    velocity_energy=_format_float(row.get("mean_velocity_energy", ""), digits=6),
                    acceleration_energy=_format_float(row.get("mean_acceleration_energy", ""), digits=6),
                    curvature_winner=row.get("curvature_winner", ""),
                    ablation_winner_family=row.get("ablation_winner_family", ""),
                    curvature_match=row.get("curvature_matches_ablation", ""),
                    distribution_winner=row.get("heuristic_distribution_winner", ""),
                    distribution_match=row.get("distribution_matches_ablation", ""),
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify Assembly actions with one normalized curvature score.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "ablations" / "fihard_ablation.yaml"),
        help="Experiment config used to resolve data roots and preprocessing.",
    )
    parser.add_argument(
        "--ablation-csv",
        default=str(ROOT / "results" / "fihard_ablations.csv"),
        help="CSV with CARD ablation results used to pull per-action prior winners.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Subset of files to analyze after the train/val/test split helper.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used by the split helper when split is not 'all'.",
    )
    parser.add_argument(
        "--max-files-per-action",
        type=int,
        default=0,
        help="Deterministic cap on the number of source files analyzed per action. Use 0 to disable subsampling.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Bootstrap replicates for confidence intervals.",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Two-sided CI level as alpha, so 0.05 means 95%% intervals.",
    )
    parser.add_argument(
        "--actions",
        default="",
        help="Optional comma-separated action list. Defaults to the union of config actions and ablation CSV actions.",
    )
    parser.add_argument(
        "--curvature-thresholds",
        default="",
        help=(
            "Optional fixed curvature thresholds as low_to_mid,mid_to_high. "
            "Defaults to tertiles over the analyzed actions."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "results" / "action_temporal_prior_analysis.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "results" / "action_temporal_prior_analysis.md"),
        help="Output Markdown report path.",
    )
    args = parser.parse_args()

    _log("Loading config and ablation metadata")
    cfg = _load_yaml(Path(args.config))
    ablation_rows = _best_ablation_rows(Path(args.ablation_csv))
    ablation_winners = _ablation_winners(ablation_rows)
    actions = _discover_actions(cfg, ablation_winners, args.actions)
    if not actions:
        raise RuntimeError("No actions available to analyze.")

    data_dir = _resolve_data_dir(cfg)
    prep = _extract_preprocessing(cfg)
    _log(
        f"Resolved {len(actions)} actions | split={args.split} | data_dir={data_dir} | "
        f"max_files_per_action={args.max_files_per_action} | bootstrap_samples={args.bootstrap_samples}"
    )
    fixed_thresholds = _parse_curvature_thresholds(args.curvature_thresholds)

    summary_rows: List[dict] = []
    for action_idx, action in enumerate(actions, start=1):
        files = _files_for_action(data_dir, action, split=args.split, seed=args.seed)
        original_count = len(files)
        files = _subsample_files(files, max_files=args.max_files_per_action, seed=args.seed)
        _log(
            f"Action {action_idx}/{len(actions)} '{action}' prepared | "
            f"files={len(files)}" + (f" (sampled from {original_count})" if len(files) != original_count else "")
        )
        summary_rows.append(
            _summarize_action(
                action,
                files,
                input_n=prep["input_n"],
                output_n=prep["output_n"],
                stride=prep["stride"],
                time_interp=prep["time_interp"],
                window_norm=prep["window_norm"],
                bootstrap_samples=args.bootstrap_samples,
                ci_alpha=args.ci_alpha,
                bootstrap_seed=args.seed + len(summary_rows) * 1009,
            )
        )

    if fixed_thresholds is None:
        thresholds = _calibrate_curvature_thresholds(summary_rows)
        threshold_source = "action_curvature_tertiles"
    else:
        thresholds = fixed_thresholds
        threshold_source = "fixed_cli_thresholds"
    _log(
        "Classifying curvature with "
        f"low_to_mid={thresholds[0]:.6g}, mid_to_high={thresholds[1]:.6g} ({threshold_source})"
    )
    summary_rows = _apply_curvature_classification(
        summary_rows,
        thresholds,
        threshold_source=threshold_source,
    )

    _log("Merging analysis with ablation winners")
    merged = _merge_with_ablation(summary_rows, ablation_winners)
    merged.sort(key=lambda row: str(row["action"]))
    _log(f"Writing CSV report to {args.output_csv}")
    _write_csv(Path(args.output_csv), merged)
    _log(f"Writing Markdown report to {args.output_md}")
    _write_markdown(
        Path(args.output_md),
        merged,
        split=args.split,
        input_n=prep["input_n"],
        output_n=prep["output_n"],
        stride=prep["stride"],
        data_dir=data_dir,
        max_files_per_action=args.max_files_per_action,
    )

    _log(f"Wrote CSV report to {args.output_csv}")
    _log(f"Wrote Markdown report to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
