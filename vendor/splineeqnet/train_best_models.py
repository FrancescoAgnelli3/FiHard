import argparse
import json
import math
import os
import time
from typing import Dict, Optional, Tuple

from config import DATASET_CHOICES, DatasetCfg, TrainCfg, MODEL_DEFAULT_CONFIGS
from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BEST_MODELS_PATH = os.path.join(REPO_ROOT, "out", "best_runs", "assembly", "best_models.json")
BEST_RESULTS_PATH = os.path.join(REPO_ROOT, "out", "best_runs", "results.csv")
DEFAULT_DATA_DIR = ""
DEFAULT_SAVE_ROOT = os.path.join(REPO_ROOT, "out", "best_runs", "checkpoints")

FIELDNAMES = [
    "timestamp",
    "model",
    "dataset",
    "seed",
    "input_n",
    "output_n",
    "stride",
    "batch_size",
    "hidden_size",
    "lr",
    "gradient_clip",
    "use_space",
    "bone_loss_weight",
    "velocity_loss_weight",
    "dct_keep_coeffs",
    "epochs",
    "params",
    "train_mpjpe_best",
    "train_mpjpe_norm_best",
    "validation_mpjpe_best",
    "validation_mpjpe_norm_best",
    "validation_humanmac_apd_best",
    "validation_humanmac_ade_best",
    "validation_humanmac_fde_best",
    "validation_humanmac_mmade_best",
    "validation_humanmac_mmfde_best",
    "validation_humanmac_cmd_best",
    "validation_humanmac_fid_best",
    "test_mpjpe_best",
    "test_mpjpe_norm_best",
    "test_humanmac_apd_best",
    "test_humanmac_ade_best",
    "test_humanmac_fde_best",
    "test_humanmac_mmade_best",
    "test_humanmac_mmfde_best",
    "test_humanmac_cmd_best",
    "test_humanmac_fid_best",
]


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _coerce_optional(value: object, cast):
    if value in (None, "", "None"):
        return None
    if _is_nan(value):
        return None
    try:
        if cast is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(int(value))
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes"}:
                    return True
                if lowered in {"0", "false", "no"}:
                    return False
        return cast(value)
    except (TypeError, ValueError):
        return None


def _coerce_required(value: object, cast, *, default):
    coerced = _coerce_optional(value, cast)
    return coerced if coerced is not None else default


def _dataset_file_path(base_path: str, dataset: str) -> str:
    base_dir, base_file = os.path.split(base_path)
    dataset_dir = os.path.join(base_dir, dataset) if base_dir else dataset
    return os.path.join(dataset_dir, base_file)


def _load_best_config(best_json: str, model_name: str) -> Dict[str, object]:
    normalized_name = model_name.lower()
    if best_json and os.path.exists(best_json):
        try:
            with open(best_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                if normalized_name in payload and isinstance(payload[normalized_name], dict):
                    return dict(payload[normalized_name])
                if isinstance(payload.get("model"), str) and payload.get("model", "").strip().lower() == normalized_name:
                    return dict(payload)
        except Exception:
            pass
    if normalized_name not in MODEL_DEFAULT_CONFIGS:
        raise KeyError(
            f"No default configuration registered for model '{model_name}', and no usable config was found in '{best_json}'."
        )
    return dict(MODEL_DEFAULT_CONFIGS[normalized_name])


def _build_dataset_cfg(
    best_cfg: Dict[str, object],
    *,
    data_dir: str,
    action_filter: str,
    dataset_name: str,
    wrist_indices: Tuple[int, ...],
    node_count: int,
    edge_index: Tuple[Tuple[int, int], ...],
    adjacency: Tuple[Tuple[int, ...], ...],
    window_override: Optional[int] = None,
    horizon_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    eval_batch_mult: Optional[int] = None,
) -> DatasetCfg:
    batch_size = _coerce_required(best_cfg.get("batch_size"), int, default=128)
    input_n = window_override if window_override is not None else _coerce_required(best_cfg.get("input_n"), int, default=90)
    output_n = (
        horizon_override if horizon_override is not None else _coerce_required(best_cfg.get("output_n"), int, default=10)
    )
    stride = _coerce_required(best_cfg.get("stride"), int, default=5)
    time_interp = _coerce_optional(best_cfg.get("time_interp"), int)
    window_norm = _coerce_optional(best_cfg.get("window_norm"), int)
    seed = seed_override if seed_override is not None else _coerce_required(best_cfg.get("seed"), int, default=0)
    eval_mult = eval_batch_mult if eval_batch_mult is not None else 1
    return DatasetCfg(
        data_dir=data_dir,
        action_filter=action_filter,
        input_n=input_n,
        output_n=output_n,
        stride=stride,
        time_interp=time_interp,
        window_norm=window_norm,
        batch_size=batch_size,
        eval_batch_mult=eval_mult,
        seed=seed,
        wrist_indices=wrist_indices,
        dataset=dataset_name,
        node_count=node_count,
        edge_index=edge_index,
        adjacency=adjacency,
    )


def _build_train_cfg(
    best_cfg: Dict[str, object], *, epochs_override: Optional[int] = None, model_name: Optional[str] = None
) -> TrainCfg:
    target_model = (model_name or str(best_cfg.get("model") or "")).lower()
    use_space_raw = best_cfg.get("use_space")
    use_space = bool(int(use_space_raw)) if use_space_raw not in (None, "", "None") else True

    resolved_model = model_name if model_name is not None else best_cfg.get("model")
    if resolved_model is None:
        resolved_model = "FiHard"
    hidden_default = 128
    epochs_default = 50
    epochs = (
        epochs_override
        if epochs_override is not None
        else _coerce_required(best_cfg.get("epochs"), int, default=epochs_default)
    )
    cfg = TrainCfg(
        model=str(resolved_model),
        epochs=epochs,
        lr=_coerce_required(best_cfg.get("lr"), float, default=1e-3),
        hidden_size=_coerce_required(best_cfg.get("hidden_size"), int, default=hidden_default),
        gru_layers=_coerce_required(best_cfg.get("gru_layers"), int, default=2),
        gradient_clip=_coerce_required(best_cfg.get("gradient_clip"), float, default=10.0),
        bone_loss_weight=_coerce_required(best_cfg.get("bone_loss_weight"), float, default=0.0),
        use_space=use_space,
        early_stopping_enabled=_coerce_required(best_cfg.get("early_stopping_enabled"), bool, default=True),
        early_stopping_patience=_coerce_required(best_cfg.get("early_stopping_patience"), int, default=20),
        early_stopping_min_delta=_coerce_required(best_cfg.get("early_stopping_min_delta"), float, default=1e-4),
        early_stopping_warmup=_coerce_required(best_cfg.get("early_stopping_warmup"), int, default=0),
        early_stopping_monitor=_coerce_required(best_cfg.get("early_stopping_monitor"), str, default="auto"),
    )

    cfg.dct_keep_coeffs = _coerce_optional(best_cfg.get("dct_keep_coeffs"), int)
    cfg.simlpe_norm_axis = _coerce_optional(best_cfg.get("simlpe_norm_axis"), str)
    cfg.simlpe_use_norm = _coerce_optional(best_cfg.get("simlpe_use_norm"), bool)
    cfg.simlpe_use_spatial_fc_only = _coerce_optional(best_cfg.get("simlpe_use_spatial_fc_only"), bool)
    cfg.simlpe_mix_spatial_temporal = _coerce_optional(best_cfg.get("simlpe_mix_spatial_temporal"), bool)
    cfg.simlpe_add_last_offset = _coerce_optional(best_cfg.get("simlpe_add_last_offset"), bool)
    cfg.FiHard_diffusion_epochs = _coerce_optional(best_cfg.get("FiHard_diffusion_epochs"), int)
    cfg.FiHard_k_low = _coerce_optional(best_cfg.get("FiHard_k_low"), int)
    cfg.FiHard_diffusion_steps = _coerce_optional(best_cfg.get("FiHard_diffusion_steps"), int)
    cfg.FiHard_ddim_steps = _coerce_optional(best_cfg.get("FiHard_ddim_steps"), int)
    cfg.FiHard_isotropic_noise = _coerce_optional(best_cfg.get("FiHard_isotropic_noise"), bool)
    cfg.FiHard_spatial_anisotropy = _coerce_optional(best_cfg.get("FiHard_spatial_anisotropy"), bool)
    cfg.FiHard_beta_matrix_power = _coerce_optional(best_cfg.get("FiHard_beta_matrix_power"), float)
    cfg.FiHard_beta_matrix_min_rate = _coerce_optional(
        best_cfg.get("FiHard_beta_matrix_min_rate"), float
    )
    cfg.FiHard_beta_matrix_max_rate = _coerce_optional(
        best_cfg.get("FiHard_beta_matrix_max_rate"), float
    )
    cfg.FiHard_temporal_anisotropy = _coerce_optional(
        best_cfg.get("FiHard_temporal_anisotropy"), bool
    )
    cfg.FiHard_temporal_anisotropy_q = _coerce_optional(
        best_cfg.get("FiHard_temporal_anisotropy_q"), float
    )
    cfg.FiHard_temporal_operator_type = _coerce_optional(
        best_cfg.get("FiHard_temporal_operator_type"), str
    )
    cfg.FiHard_temporal_operator_spectral_transform = _coerce_optional(
        best_cfg.get("FiHard_temporal_operator_spectral_transform"), str
    )
    cfg.FiHard_temporal_velocity_weight = _coerce_optional(
        best_cfg.get("FiHard_temporal_velocity_weight"), float
    )
    cfg.FiHard_temporal_acceleration_weight = _coerce_optional(
        best_cfg.get("FiHard_temporal_acceleration_weight"), float
    )
    cfg.FiHard_temporal_jerk_weight = _coerce_optional(
        best_cfg.get("FiHard_temporal_jerk_weight"), float
    )
    cfg.FiHard_temporal_anisotropy_learned_from_history = _coerce_optional(
        best_cfg.get("FiHard_temporal_anisotropy_learned_from_history"), bool
    )
    cfg.FiHard_temporal_anisotropy_history_dim = _coerce_optional(
        best_cfg.get("FiHard_temporal_anisotropy_history_dim"), int
    )
    cfg.FiHard_temporal_anisotropy_delta_max_abs = _coerce_optional(
        best_cfg.get("FiHard_temporal_anisotropy_delta_max_abs"), float
    )
    cfg.FiHard_node_covariance_type = _coerce_optional(
        best_cfg.get("FiHard_node_covariance_type"), str
    )
    cfg.FiHard_mobility_palm_var = _coerce_optional(best_cfg.get("FiHard_mobility_palm_var"), float)
    cfg.FiHard_mobility_depth1_var = _coerce_optional(best_cfg.get("FiHard_mobility_depth1_var"), float)
    cfg.FiHard_mobility_depth2_var = _coerce_optional(best_cfg.get("FiHard_mobility_depth2_var"), float)
    cfg.FiHard_mobility_depth3plus_var = _coerce_optional(
        best_cfg.get("FiHard_mobility_depth3plus_var"), float
    )
    cfg.FiHard_dhalf_gamma = _coerce_optional(best_cfg.get("FiHard_dhalf_gamma"), float)
    cfg.FiHard_learnable_dhalf = _coerce_optional(best_cfg.get("FiHard_learnable_dhalf"), bool)
    cfg.FiHard_graph_laplacian_alpha = _coerce_optional(
        best_cfg.get("FiHard_graph_laplacian_alpha"), float
    )
    cfg.FiHard_graph_laplacian_beta = _coerce_optional(
        best_cfg.get("FiHard_graph_laplacian_beta"), float
    )
    cfg.FiHard_graph_laplacian_normalized = _coerce_optional(
        best_cfg.get("FiHard_graph_laplacian_normalized"), bool
    )
    cfg.FiHard_denoiser_dim = _coerce_optional(best_cfg.get("FiHard_denoiser_dim"), int)
    cfg.FiHard_denoiser_depth = _coerce_optional(best_cfg.get("FiHard_denoiser_depth"), int)
    cfg.FiHard_denoiser_heads = _coerce_optional(best_cfg.get("FiHard_denoiser_heads"), int)
    cfg.FiHard_dropout = _coerce_optional(best_cfg.get("FiHard_dropout"), float)
    cfg.FiHard_freeze_coarse = _coerce_optional(best_cfg.get("FiHard_freeze_coarse"), bool)
    cfg.FiHard_diffusion_coarse_warmup_epochs = _coerce_optional(
        best_cfg.get("FiHard_diffusion_coarse_warmup_epochs"), int
    )
    cfg.FiHard_cond_use_history = _coerce_optional(best_cfg.get("FiHard_cond_use_history"), bool)
    cfg.FiHard_cond_use_coarse = _coerce_optional(best_cfg.get("FiHard_cond_use_coarse"), bool)
    cfg.FiHard_allow_no_conditioning = _coerce_optional(
        best_cfg.get("FiHard_allow_no_conditioning"), bool
    )
    cfg.FiHard_diffusion_only = _coerce_optional(
        best_cfg.get("FiHard_diffusion_only"), bool
    )
    cfg.FiHard_use_mamp_condition = _coerce_optional(best_cfg.get("FiHard_use_mamp_condition"), bool)
    cfg.FiHard_use_mamp_condition_coarse = _coerce_optional(best_cfg.get("FiHard_use_mamp_condition_coarse"), bool)
    cfg.FiHard_mamp_checkpoint = _coerce_optional(best_cfg.get("FiHard_mamp_checkpoint"), str)
    cfg.FiHard_mamp_config = _coerce_optional(best_cfg.get("FiHard_mamp_config"), str)
    cfg.FiHard_mamp_repo_root = _coerce_optional(best_cfg.get("FiHard_mamp_repo_root"), str)
    cfg.FiHard_mamp_mask_ratio = _coerce_optional(best_cfg.get("FiHard_mamp_mask_ratio"), float)
    cfg.FiHard_mamp_motion_aware_tau = _coerce_optional(best_cfg.get("FiHard_mamp_motion_aware_tau"), float)
    cfg.FiHard_mpjpe_weight = _coerce_optional(best_cfg.get("FiHard_mpjpe_weight"), float)
    cfg.FiHard_coarse_target_lowpass_only = _coerce_optional(
        best_cfg.get("FiHard_coarse_target_lowpass_only"), bool
    )
    cfg.FiHard_graph_laplacian_tau = _coerce_optional(best_cfg.get("FiHard_graph_laplacian_tau"), float)
    cfg.FiHard_covariance_jitter = _coerce_optional(best_cfg.get("FiHard_covariance_jitter"), float)
    return cfg


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_results_row(results_csv: str, row: Dict[str, object]) -> None:
    write_header = not os.path.exists(results_csv)
    if not write_header:
        with open(results_csv, "r", newline="") as existing:
            existing_header = existing.readline().strip()
        if not existing_header:
            write_header = True

    with open(results_csv, "a", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for key in FIELDNAMES:
            if row.get(key) is None:
                row[key] = ""
        writer.writerow(row)


def _append_window_horizon(path: str, window: int, horizon: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_w{window}_h{horizon}{ext}"


def _prepare_result_row(ds_cfg: DatasetCfg, train_cfg: TrainCfg, metrics: Dict[str, float]) -> Dict[str, object]:
    base = {
        "timestamp": int(time.time()),
        "model": train_cfg.model,
        "dataset": ds_cfg.dataset,
        "seed": ds_cfg.seed,
        "input_n": ds_cfg.input_n,
        "output_n": ds_cfg.output_n,
        "stride": ds_cfg.stride,
        "batch_size": ds_cfg.batch_size,
        "hidden_size": train_cfg.hidden_size,
        "lr": train_cfg.lr,
        "gradient_clip": train_cfg.gradient_clip,
        "use_space": int(train_cfg.use_space),
        "bone_loss_weight": train_cfg.bone_loss_weight,
        "velocity_loss_weight": train_cfg.velocity_loss_weight,
        "dct_keep_coeffs": train_cfg.dct_keep_coeffs,
        "epochs": train_cfg.epochs,
        "params": metrics.get("params", 0),
        "train_mpjpe_best": metrics.get("train_mpjpe_best"),
        "train_mpjpe_norm_best": metrics.get("train_mpjpe_norm_best"),
        "validation_mpjpe_best": metrics.get("validation_mpjpe_best"),
        "validation_mpjpe_norm_best": metrics.get("validation_mpjpe_norm_best"),
        "validation_humanmac_apd_best": metrics.get("validation_humanmac_apd_best"),
        "validation_humanmac_ade_best": metrics.get("validation_humanmac_ade_best"),
        "validation_humanmac_fde_best": metrics.get("validation_humanmac_fde_best"),
        "validation_humanmac_mmade_best": metrics.get("validation_humanmac_mmade_best"),
        "validation_humanmac_mmfde_best": metrics.get("validation_humanmac_mmfde_best"),
        "validation_humanmac_cmd_best": metrics.get("validation_humanmac_cmd_best"),
        "validation_humanmac_fid_best": metrics.get("validation_humanmac_fid_best"),
        "test_mpjpe_best": metrics.get("test_mpjpe_best"),
        "test_mpjpe_norm_best": metrics.get("test_mpjpe_norm_best"),
        "test_humanmac_apd_best": metrics.get("test_humanmac_apd_best"),
        "test_humanmac_ade_best": metrics.get("test_humanmac_ade_best"),
        "test_humanmac_fde_best": metrics.get("test_humanmac_fde_best"),
        "test_humanmac_mmade_best": metrics.get("test_humanmac_mmade_best"),
        "test_humanmac_mmfde_best": metrics.get("test_humanmac_mmfde_best"),
        "test_humanmac_cmd_best": metrics.get("test_humanmac_cmd_best"),
        "test_humanmac_fid_best": metrics.get("test_humanmac_fid_best"),
    }

    row: Dict[str, object] = {}
    for field in FIELDNAMES:
        value = base.get(field)
        row[field] = "" if value is None else value
    return row


def main():
    ap = argparse.ArgumentParser(description="Train, validate, and test the best configuration for one or more models.")
    ap.add_argument(
        "--model",
        nargs="+",
        default="FiHard",
        help="One or more model names (space- or comma-separated) whose best configuration should be trained.",
    )
    ap.add_argument("--dataset", default="assembly", choices=DATASET_CHOICES, help="Dataset to use for training the best model.")
    ap.add_argument("--best-json", default=BEST_MODELS_PATH, help="Path to JSON file with best configurations.")
    ap.add_argument("--results-csv", default=BEST_RESULTS_PATH, help="CSV file where evaluation results will be logged.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root directory containing dataset files.")
    ap.add_argument("--action-filter", default="", help="Action substring filter for dataset files.")
    ap.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="Directory to store model artifacts.")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed for dataset/build pipeline.")
    ap.add_argument("--eval-batch-mult", type=int, default=1, help="Multiplier for validation/test batch size.")
    ap.add_argument("--epochs", type=int, default=100, help="Override the number of training epochs from the best config.")
    ap.add_argument("--log-wandb", action="store_true", help="Enable logging to Weights & Biases.")
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    ap.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    ap.add_argument("--wandb-run-prefix", type=str, default=None, help="Optional run name prefix for W&B.")
    ap.add_argument("--save-eval-examples", action="store_true", help="When set, persist evaluation examples (merged_pred/merged_tgt tensors) during validation/testing.")
    ap.add_argument(
        "--eval-examples-dir",
        default="",
        help="Optional directory where evaluation sample bundles should be written.",
    )
    ap.add_argument(
        "--eval-examples-path",
        default="",
        help="Optional explicit output path for the evaluation sample bundle.",
    )
    ap.add_argument("--save-coarse-model", action="store_true", help="When set, save/load the FiHard coarse model weights for this configuration.")
    ap.add_argument("--save-best-model", action="store_true", help="When set, persist the best model checkpoint into the examples directory.")
    ap.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Optional explicit checkpoint output path used when --save-best-model is enabled.",
    )
    ap.add_argument(
        "--final-checkpoint-path",
        type=str,
        default="",
        help="Optional explicit output path used to persist the final model state after training.",
    )
    ap.add_argument("--window", type=int,default=None, help="Optional override for the input window length (input_n).")
    ap.add_argument("--horizon",type=int,default=None,help="Optional override for the prediction horizon length (output_n).")
    ap.add_argument("--FiHard-mamp-checkpoint", type=str, default="", help="Path to pretrained MAMP checkpoint.")
    ap.add_argument("--FiHard-mamp-config", type=str, default="", help="Optional MAMP YAML with model_args.")
    ap.add_argument("--FiHard-mamp-repo-root", type=str, default="", help="Path to MAMP repo root.")
    ap.add_argument("--FiHard-mamp-mask-ratio", type=float, default=0.0, help="Mask ratio for MAMP encoder (0 for deterministic full context).")
    ap.add_argument("--FiHard-mamp-motion-aware-tau", type=float, default=0.80, help="Motion-aware tau for MAMP encoder.")
    ap.add_argument("--FiHard-use-mamp-condition", action="store_true", help="Enable MAMP conditioning in two-stage diffusion.")
    ap.add_argument("--FiHard-use-mamp-condition-coarse", action="store_true", help="Enable coarse-prediction MAMP conditioning in two-stage diffusion.")
    ap.add_argument("--FiHard-mamp-only-conditioning", action="store_true", help="Use only MAMP conditioning tokens (disable history/coarse conditioning).")
    ap.add_argument("--num-candidates", type=int, default=10, help="Best-of-k and HumanMAC candidate count for evaluation metrics.")
    ap.add_argument("--humanmac-multimodal-threshold", type=float, default=0.5, help="HumanMAC multimodal grouping threshold.")
    ap.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        choices=("validation", "test"),
        help="Evaluation split used for final reported metrics.",
    )
    args = ap.parse_args()

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    data_dir = args.data_dir or metadata.get("default_dir", "")
    action_filter = args.action_filter or metadata.get("default_action_filter", "")
    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if not default_wrist_indices:
        raise SystemExit("Dataset metadata must define default wrist indices.")
    wrist_indices = tuple(int(idx) for idx in default_wrist_indices)

    best_json_path = _dataset_file_path(args.best_json, dataset_name)
    results_csv_base = _dataset_file_path(args.results_csv, dataset_name)
    if args.action_filter:
        results_csv_base = results_csv_base.replace(".csv", f"_{args.action_filter}.csv")
    save_root = os.path.join(args.save_root, dataset_name)

    if not os.path.exists(best_json_path) and os.path.exists(args.best_json):
        best_json_path = args.best_json

    # Parse model names: support space- and comma-separated inputs
    raw_models = args.model if isinstance(args.model, list) else [str(args.model)]
    model_names = []
    for token in raw_models:
        model_names.extend([m.strip() for m in str(token).split(",") if m.strip()])

    if not model_names:
        raise SystemExit("No valid model names provided to --model")

    _ensure_dir(save_root)

    results_csv_path: Optional[str] = None

    any_failed = False
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

            train_cfg = _build_train_cfg(best_cfg, epochs_override=args.epochs, model_name=model_name)
            # Ensure the model field reflects the current target model
            train_cfg.model = model_name
            train_cfg.save_eval_examples = bool(args.save_eval_examples)
            if args.eval_examples_path:
                train_cfg.eval_examples_path = str(args.eval_examples_path)
            elif args.eval_examples_dir:
                train_cfg.eval_examples_dir = str(args.eval_examples_dir)
            else:
                train_cfg.eval_examples_dir = os.path.join(save_root, "eval_examples", model_name)
            train_cfg.save_coarse_model = bool(args.save_coarse_model)
            if args.FiHard_use_mamp_condition:
                train_cfg.FiHard_use_mamp_condition = True
            if args.FiHard_use_mamp_condition_coarse:
                train_cfg.FiHard_use_mamp_condition_coarse = True
            if args.FiHard_mamp_checkpoint:
                train_cfg.FiHard_use_mamp_condition = True
                train_cfg.FiHard_mamp_checkpoint = str(args.FiHard_mamp_checkpoint)
            if args.FiHard_mamp_config:
                train_cfg.FiHard_mamp_config = str(args.FiHard_mamp_config)
            if args.FiHard_mamp_repo_root:
                train_cfg.FiHard_mamp_repo_root = str(args.FiHard_mamp_repo_root)
            train_cfg.FiHard_mamp_mask_ratio = float(args.FiHard_mamp_mask_ratio)
            train_cfg.FiHard_mamp_motion_aware_tau = float(args.FiHard_mamp_motion_aware_tau)
            if args.FiHard_mamp_only_conditioning:
                train_cfg.FiHard_use_mamp_condition = True
                train_cfg.FiHard_cond_use_history = False
                train_cfg.FiHard_cond_use_coarse = False

            train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
            train_loader, val_loader, test_loader = make_loaders(
                train_dataset, val_dataset, test_dataset, ds_cfg.batch_size, ds_cfg.seed, ds_cfg.eval_batch_mult
            )
            eval_loader = test_loader if str(args.eval_split).lower() == "test" else val_loader

            if results_csv_path is None:
                results_csv_path = _append_window_horizon(results_csv_base, ds_cfg.input_n, ds_cfg.output_n)
                _ensure_dir(os.path.dirname(results_csv_path) or ".")
            else:
                expected_path = _append_window_horizon(results_csv_base, ds_cfg.input_n, ds_cfg.output_n)
                if expected_path != results_csv_path:
                    raise ValueError(
                        "Inconsistent dataset window/horizon detected across models; "
                        "cannot log to a shared results file with differing configurations."
                    )

            metrics = run_experiment(
                ds=ds_cfg,
                train_cfg=train_cfg,
                save_root=save_root,
                train_loader=train_loader,
                val_loader=eval_loader,
                test_loader=None,
                log_wandb=args.log_wandb or bool(args.wandb_project),
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_run_prefix=args.wandb_run_prefix,
                save_best_model=args.save_best_model,
                best_model_path_override=(args.checkpoint_path.strip() or None),
                final_model_path_override=(args.final_checkpoint_path.strip() or None),
                num_candidates=max(1, int(args.num_candidates)),
                FiHard_eval_oracle_mpjpe=(model_name.strip().lower() == "fihard"),
                compute_humanmac_metrics=True,
                humanmac_multimodal_threshold=float(args.humanmac_multimodal_threshold),
            )
            if metrics is None:
                raise RuntimeError("Training failed to produce metrics.")

            row = _prepare_result_row(ds_cfg, train_cfg, metrics)
            _write_results_row(results_csv_path, row)
            print(f"Recorded results for {model_name}:", row)

        except Exception as e:
            any_failed = True
            print(f"Error while training model '{model_name}': {e}")

    if any_failed:
        raise SystemExit("One or more models failed during training. Check logs above.")


if __name__ == "__main__":
    main()
