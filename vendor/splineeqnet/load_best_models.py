import argparse
import csv
import json
import os
from typing import Dict, Optional, Tuple

from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment
from train_best_models import (
    _build_dataset_cfg,
    _build_train_cfg,
    _dataset_file_path,
    _ensure_dir,
    _load_best_config,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BEST_MODELS_PATH = os.path.join(REPO_ROOT, "out", "best_runs", "assembly", "best_models.json")
DEFAULT_DATA_DIR = ""
DEFAULT_CHECKPOINT_ROOT = os.path.join(REPO_ROOT, "examples", "models")
DEFAULT_RESULTS_CSV = os.path.join(REPO_ROOT, "out", "eval", "load_best_models_metrics.csv")


def _normalize_model_name(name: str) -> str:
    return name.strip().lower()


def _default_checkpoint_path(
    model_name: str, dataset: str, action_filter: str, checkpoint_root: str
) -> str:
    action_segment = action_filter or "all"
    action_segment = action_segment.replace(os.sep, "_").replace(" ", "_")
    normalized_model = _normalize_model_name(model_name)
    model_file = f"{normalized_model}_{dataset}_{action_segment}.pt"
    return os.path.join(checkpoint_root, model_file)


def _write_results_csv(results_csv: str, rows: list[Dict[str, object]]) -> None:
    if not rows:
        return
    _ensure_dir(os.path.dirname(results_csv))
    fieldnames = [
        "model",
        "dataset",
        "action_filter",
        "checkpoint",
        "num_candidates",
        "humanmac_multimodal_threshold",
        "test_mpjpe_best",
        "test_mpjpe_norm_best",
        "test_samples",
        "test_humanmac_apd_best",
        "test_humanmac_ade_best",
        "test_humanmac_fde_best",
        "test_humanmac_mmade_best",
        "test_humanmac_mmfde_best",
        "test_humanmac_cmd_best",
        "test_humanmac_fid_best",
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> Dict[str, Dict[str, object]]:
    ap = argparse.ArgumentParser(description="Load best model checkpoints and evaluate on test set.")
    ap.add_argument(
        "--model",
        nargs="+",
        default="card",
        help="One or more model names (space- or comma-separated) whose best checkpoint should be evaluated.",
    )
    ap.add_argument("--dataset", default="assembly", help="Dataset to use for evaluation.")
    ap.add_argument("--best-json", default=BEST_MODELS_PATH, help="Path to JSON file with best configurations.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root directory containing dataset files.")
    ap.add_argument("--action-filter", default="", help="Action substring filter for dataset files.")
    ap.add_argument("--checkpoint", default="", help="Optional explicit checkpoint path to load.")
    ap.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT, help="Root folder for saved checkpoints.")
    ap.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, help="CSV path for saved evaluation metrics.")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed for dataset/build pipeline.")
    ap.add_argument("--eval-batch-mult", type=int, default=1, help="Multiplier for evaluation batch size.")
    ap.add_argument("--save-eval-examples", action="store_true", help="When set, persist evaluation examples (merged_pred/merged_tgt tensors) during evaluation.")
    ap.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help=(
            "Shared candidate count used by both card diffusion best-of-k selection "
            "and HumanMAC candidate evaluation."
        ),
    )
    ap.add_argument(
        "--humanmac-multimodal-threshold",
        type=float,
        default=0.5,
        help="Start-pose distance threshold used to build HumanMAC-style multimodal groups.",
    )
    ap.add_argument("--window", type=int, default=None, help="Optional override for the input window length (input_n).")
    ap.add_argument("--horizon", type=int, default=None, help="Optional override for the prediction horizon length (output_n).")
    args = ap.parse_args()
    num_candidates = max(1, int(args.num_candidates))

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    data_dir = args.data_dir or metadata.get("default_dir", "")
    action_filter = args.action_filter or metadata.get("default_action_filter", "")
    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if not default_wrist_indices:
        raise SystemExit("Dataset metadata must define default wrist indices.")
    wrist_indices = tuple(int(idx) for idx in default_wrist_indices)

    best_json_path = _dataset_file_path(args.best_json, dataset_name)
    if not os.path.exists(best_json_path) and os.path.exists(args.best_json):
        best_json_path = args.best_json

    raw_models = args.model if isinstance(args.model, list) else [str(args.model)]
    model_names = []
    for token in raw_models:
        model_names.extend([m.strip() for m in str(token).split(",") if m.strip()])

    if not model_names:
        raise SystemExit("No valid model names provided to --model")

    _ensure_dir(args.checkpoint_root)

    any_failed = False
    results_by_model: Dict[str, Dict[str, object]] = {}
    csv_rows: list[Dict[str, object]] = []
    for model_name in model_names:
        try:
            best_cfg = _load_best_config(best_json_path, model_name)

            ds_cfg = _build_dataset_cfg(
                best_cfg,
                data_dir=data_dir,
                action_filter=action_filter,
                dataset_name=dataset_name,
                wrist_indices=wrist_indices,
                node_count=int(metadata.get("node_count", 42)),
                edge_index=tuple(metadata.get("edge_index", ())),
                adjacency=tuple(metadata.get("adjacency", ())),
                window_override=args.window,
                horizon_override=args.horizon,
                seed_override=args.seed,
                eval_batch_mult=args.eval_batch_mult,
            )

            train_cfg = _build_train_cfg(best_cfg, epochs_override=0, model_name=model_name)
            train_cfg.model = model_name
            train_cfg.epochs = 0
            train_cfg.card_diffusion_epochs = 0
            train_cfg.save_eval_examples = bool(args.save_eval_examples)

            train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
            train_loader, _, test_loader = make_loaders(
                train_dataset, val_dataset, test_dataset, ds_cfg.batch_size, ds_cfg.seed, ds_cfg.eval_batch_mult
            )

            checkpoint_path = args.checkpoint.strip()
            if not checkpoint_path:
                checkpoint_path = _default_checkpoint_path(
                    model_name,
                    ds_cfg.dataset,
                    ds_cfg.action_filter or "",
                    args.checkpoint_root,
                )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

            model_name_clean = model_name.strip().lower()
            collect_card_tries = (
                model_name_clean == "card" and num_candidates > 1
            )
            metrics = run_experiment(
                ds=ds_cfg,
                train_cfg=train_cfg,
                save_root=".",
                train_loader=train_loader,
                val_loader=test_loader,
                test_loader=None,
                log_wandb=False,
                load_model_path=checkpoint_path,
                card_eval_phase="diffusion",
                num_candidates=num_candidates,
                card_eval_collect_all=collect_card_tries,
                card_eval_oracle_mpjpe=True,
                compute_humanmac_metrics=True,
                humanmac_multimodal_threshold=float(args.humanmac_multimodal_threshold),
            )
            if metrics is None:
                raise RuntimeError("Evaluation failed to produce metrics.")

            test_mpjpe = metrics.get("test_mpjpe_best", metrics.get("validation_mpjpe_best"))
            test_mpjpe_norm = metrics.get("test_mpjpe_norm_best", metrics.get("validation_mpjpe_norm_best"))
            test_samples = metrics.get("test_samples", metrics.get("validation_samples"))
            test_humanmac_apd = metrics.get("test_humanmac_apd_best", metrics.get("validation_humanmac_apd_best"))
            test_humanmac_ade = metrics.get("test_humanmac_ade_best", metrics.get("validation_humanmac_ade_best"))
            test_humanmac_fde = metrics.get("test_humanmac_fde_best", metrics.get("validation_humanmac_fde_best"))
            test_humanmac_mmade = metrics.get("test_humanmac_mmade_best", metrics.get("validation_humanmac_mmade_best"))
            test_humanmac_mmfde = metrics.get("test_humanmac_mmfde_best", metrics.get("validation_humanmac_mmfde_best"))
            test_humanmac_cmd = metrics.get("test_humanmac_cmd_best", metrics.get("validation_humanmac_cmd_best"))
            test_humanmac_fid = metrics.get("test_humanmac_fid_best", metrics.get("validation_humanmac_fid_best"))
            model_result = {
                "model": model_name,
                "dataset": ds_cfg.dataset,
                "action_filter": ds_cfg.action_filter,
                "checkpoint": checkpoint_path,
                "num_candidates": num_candidates,
                "humanmac_multimodal_threshold": float(args.humanmac_multimodal_threshold),
                "test_mpjpe_best": test_mpjpe,
                "test_mpjpe_norm_best": test_mpjpe_norm,
                "test_samples": test_samples,
                "test_humanmac_apd_best": test_humanmac_apd,
                "test_humanmac_ade_best": test_humanmac_ade,
                "test_humanmac_fde_best": test_humanmac_fde,
                "test_humanmac_mmade_best": test_humanmac_mmade,
                "test_humanmac_mmfde_best": test_humanmac_mmfde,
                "test_humanmac_cmd_best": test_humanmac_cmd,
                "test_humanmac_fid_best": test_humanmac_fid,
            }
            results_by_model[model_name] = model_result
            csv_rows.append(model_result)
            print(f"[{model_name}] {json.dumps(model_result, sort_keys=True)}")
            if collect_card_tries:
                per_try = metrics.get("test_mpjpe_by_try", metrics.get("validation_mpjpe_by_try")) or []
                per_try_norm = metrics.get("test_mpjpe_norm_by_try", metrics.get("validation_mpjpe_norm_by_try")) or []
                if per_try:
                    for idx, mpjpe in enumerate(per_try, start=1):
                        mpjpe_norm = per_try_norm[idx - 1] if idx - 1 < len(per_try_norm) else None
                        print(
                            f"[{model_name}] try {idx}/{len(per_try)} test_mpjpe={mpjpe} | "
                            f"test_mpjpe_norm={mpjpe_norm}"
                        )

        except Exception as exc:
            any_failed = True
            print(f"Error while evaluating model '{model_name}': {exc}")

    if any_failed:
        raise SystemExit("One or more models failed during evaluation. Check logs above.")
    _write_results_csv(args.results_csv, csv_rows)
    print(f"Saved metrics CSV to {args.results_csv}")
    return results_by_model


if __name__ == "__main__":
    main()
