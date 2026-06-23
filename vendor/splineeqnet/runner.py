import hashlib
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DatasetCfg, TrainCfg
from data import resolve_card_hand_graph_metadata
from train_utils import train


def set_global_seed(seed: int) -> None:
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int % (2**32))
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def _prepare_wandb_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool):
            payload[key] = float(value)
            continue
        if isinstance(value, (int, float)):
            val = float(value)
            if math.isfinite(val):
                payload[key] = val
    return payload


def run_experiment(
    ds: DatasetCfg,
    train_cfg: TrainCfg,
    *,
    save_root: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    log_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_prefix: Optional[str] = None,
    save_best_model: bool = False,
    best_model_path_override: Optional[str] = None,
    final_model_path_override: Optional[str] = None,
    load_model_path: Optional[str] = None,
    card_eval_phase: Optional[str] = None,
    card_eval_best_of_k: Optional[int] = None,
    card_eval_collect_all: bool = False,
    card_eval_oracle_mpjpe: bool = False,
    compute_humanmac_metrics: bool = False,
    num_candidates: Optional[int] = None,
    humanmac_num_candidates: Optional[int] = None,
    humanmac_multimodal_threshold: float = 0.5,
) -> Optional[Dict[str, float]]:
    set_global_seed(ds.seed)
    card_graph_meta = resolve_card_hand_graph_metadata(
        ds.dataset,
        tuple(int(idx) for idx in ds.wrist_indices),
    )

    model_key = str(train_cfg.model).strip().lower()

    tag = (
        f"dataset={ds.dataset}__model={train_cfg.model}__hs={train_cfg.hidden_size}__gl={train_cfg.gru_layers}__"
        f"lr={train_cfg.lr}__gc={train_cfg.gradient_clip}__bs={ds.batch_size}__"
        f"in={ds.input_n}__out={ds.output_n}__st={ds.stride}__seed={ds.seed}"
    )
    if train_cfg.dct_keep_coeffs is not None:
        tag += f"__dctk={int(train_cfg.dct_keep_coeffs)}"
    if train_cfg.simlpe_norm_axis is not None:
        tag += f"__simnorm={train_cfg.simlpe_norm_axis}"
    if train_cfg.simlpe_use_norm is not None:
        tag += f"__simusenorm={int(train_cfg.simlpe_use_norm)}"
    if train_cfg.simlpe_use_spatial_fc_only is not None:
        tag += f"__simspfc={int(train_cfg.simlpe_use_spatial_fc_only)}"
    if train_cfg.simlpe_mix_spatial_temporal is not None:
        tag += f"__simmix={int(train_cfg.simlpe_mix_spatial_temporal)}"
    if train_cfg.simlpe_add_last_offset is not None:
        tag += f"__simoffset={int(train_cfg.simlpe_add_last_offset)}"
    if train_cfg.card_diffusion_epochs is not None:
        tag += f"__twepochs={int(train_cfg.card_diffusion_epochs)}"
    if train_cfg.card_k_low is not None:
        tag += f"__twk={int(train_cfg.card_k_low)}"
    if train_cfg.card_diffusion_steps is not None:
        tag += f"__twsteps={int(train_cfg.card_diffusion_steps)}"
    if train_cfg.card_ddim_steps is not None:
        tag += f"__twddim={int(train_cfg.card_ddim_steps)}"
    if train_cfg.card_spatial_anisotropy is not None:
        tag += f"__twspaniso={int(train_cfg.card_spatial_anisotropy)}"
    if train_cfg.card_temporal_anisotropy is not None:
        tag += f"__twtempaniso={int(train_cfg.card_temporal_anisotropy)}"
    if train_cfg.card_temporal_anisotropy_q is not None:
        tag += f"__twtempq={float(train_cfg.card_temporal_anisotropy_q)}"
    if train_cfg.card_temporal_operator_type is not None:
        tag += f"__twtempop={str(train_cfg.card_temporal_operator_type)}"
    if train_cfg.card_temporal_operator_spectral_transform is not None:
        tag += f"__twtempxfm={str(train_cfg.card_temporal_operator_spectral_transform)}"
    if train_cfg.card_temporal_velocity_weight is not None:
        tag += f"__twtempvw={float(train_cfg.card_temporal_velocity_weight)}"
    if train_cfg.card_temporal_acceleration_weight is not None:
        tag += f"__twtempaw={float(train_cfg.card_temporal_acceleration_weight)}"
    if train_cfg.card_temporal_jerk_weight is not None:
        tag += f"__twtempjw={float(train_cfg.card_temporal_jerk_weight)}"
    if train_cfg.card_temporal_anisotropy_learned_from_history is not None:
        tag += f"__twtemphist={int(train_cfg.card_temporal_anisotropy_learned_from_history)}"
    if train_cfg.card_coarse_target_lowpass_only is not None:
        tag += f"__twlpgt={int(train_cfg.card_coarse_target_lowpass_only)}"
    if train_cfg.card_diffusion_only is not None:
        tag += f"__twdiffonly={int(train_cfg.card_diffusion_only)}"

    config = {
        "batch_size": int(ds.batch_size),
        "input_n": int(ds.input_n),
        "output_n": int(ds.output_n),
        "learning_rate": float(train_cfg.lr),
        "train_epoches": int(train_cfg.epochs),
        "hidden_size": int(train_cfg.hidden_size),
        "gru_layers": int(train_cfg.gru_layers),
        "gradient_clip": float(train_cfg.gradient_clip),
        "node_num": int(ds.node_count),
        "use_space": bool(train_cfg.use_space),
        "velocity_loss_weight": float(train_cfg.velocity_loss_weight),
        "save_eval_examples": bool(train_cfg.save_eval_examples),
        "log_gcn_stats": False,
        "log_wandb": bool(log_wandb),
        "dataset": ds.dataset,
        "action_filter": ds.action_filter,
        "edge_index": tuple(ds.edge_index),
        "adjacency": tuple(tuple(int(val) for val in row) for row in ds.adjacency) if ds.adjacency else tuple(),
        "card_wrist_index": int(card_graph_meta["wrist_index"]),
        "card_links": tuple(
            (int(pair[0]), int(pair[1])) for pair in tuple(card_graph_meta["links"])
        ),
        "early_stopping_enabled": bool(train_cfg.early_stopping_enabled),
        "early_stopping_patience": int(train_cfg.early_stopping_patience),
        "early_stopping_min_delta": float(train_cfg.early_stopping_min_delta),
        "early_stopping_warmup": int(train_cfg.early_stopping_warmup),
        "early_stopping_monitor": str(train_cfg.early_stopping_monitor),
    }

    if train_cfg.eval_examples_path:
        config["eval_examples_path"] = str(train_cfg.eval_examples_path)
    elif train_cfg.eval_examples_dir:
        config["eval_examples_dir"] = str(train_cfg.eval_examples_dir)

    if save_best_model:
        examples_models_dir = os.path.join(os.path.dirname(__file__), "examples", "models")
        os.makedirs(examples_models_dir, exist_ok=True)
        if best_model_path_override:
            config["best_model_path"] = str(best_model_path_override)
        else:
            action_segment = (ds.action_filter or "all").replace(os.sep, "_").replace(" ", "_")
            model_file = f"{train_cfg.model}_{ds.dataset}_{action_segment}.pt"
            config["best_model_path"] = os.path.join(examples_models_dir, model_file)
        config["best_model_tag"] = tag
    if final_model_path_override:
        config["final_model_path"] = str(final_model_path_override)
    if load_model_path:
        config["load_model_path"] = str(load_model_path)
    if card_eval_phase:
        config["card_eval_phase"] = str(card_eval_phase)
    resolved_num_candidates = None
    if num_candidates is not None:
        resolved_num_candidates = max(1, int(num_candidates))
    elif card_eval_best_of_k is not None:
        resolved_num_candidates = max(1, int(card_eval_best_of_k))
    elif humanmac_num_candidates is not None:
        resolved_num_candidates = max(1, int(humanmac_num_candidates))
    if resolved_num_candidates is not None:
        config["num_candidates"] = resolved_num_candidates
        config["card_eval_best_of_k"] = resolved_num_candidates
    if card_eval_collect_all:
        config["card_eval_collect_all"] = True
    if card_eval_oracle_mpjpe:
        config["card_eval_oracle_mpjpe"] = True
    if compute_humanmac_metrics:
        config["compute_humanmac_metrics"] = True
        if resolved_num_candidates is not None:
            config["humanmac_num_candidates"] = resolved_num_candidates
        config["humanmac_multimodal_threshold"] = float(humanmac_multimodal_threshold)

    if train_cfg.save_coarse_model:
        examples_models_dir = os.path.join(os.path.dirname(__file__), "examples", "models")
        os.makedirs(examples_models_dir, exist_ok=True)
        action_segment = (ds.action_filter or "all").replace(os.sep, "_").replace(" ", "_")
        tag_hash = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
        model_file = f"{train_cfg.model}_coarse_{ds.dataset}_{action_segment}_{tag_hash}.pt"
        config["coarse_model_path"] = os.path.join(examples_models_dir, model_file)
        config["coarse_model_tag"] = tag
        config["save_coarse_model"] = True

    if train_cfg.dct_keep_coeffs is not None:
        config["dct_keep_coeffs"] = int(train_cfg.dct_keep_coeffs)
    if train_cfg.simlpe_norm_axis is not None:
        config["simlpe_norm_axis"] = str(train_cfg.simlpe_norm_axis)
    if train_cfg.simlpe_use_norm is not None:
        config["simlpe_use_norm"] = bool(train_cfg.simlpe_use_norm)
    if train_cfg.simlpe_use_spatial_fc_only is not None:
        config["simlpe_spatial_fc_only"] = bool(train_cfg.simlpe_use_spatial_fc_only)
    if train_cfg.simlpe_mix_spatial_temporal is not None:
        config["simlpe_mix_spatial_temporal"] = bool(train_cfg.simlpe_mix_spatial_temporal)
    if train_cfg.simlpe_add_last_offset is not None:
        config["simlpe_add_last_offset"] = bool(train_cfg.simlpe_add_last_offset)

    card_fields = [
        ("card_diffusion_epochs", train_cfg.card_diffusion_epochs, int),
        ("card_k_low", train_cfg.card_k_low, int),
        ("card_diffusion_steps", train_cfg.card_diffusion_steps, int),
        ("card_ddim_steps", train_cfg.card_ddim_steps, int),
        ("card_isotropic_noise", train_cfg.card_isotropic_noise, bool),
        ("card_spatial_anisotropy", train_cfg.card_spatial_anisotropy, bool),
        ("card_beta_matrix_power", train_cfg.card_beta_matrix_power, float),
        ("card_beta_matrix_min_rate", train_cfg.card_beta_matrix_min_rate, float),
        ("card_beta_matrix_max_rate", train_cfg.card_beta_matrix_max_rate, float),
        ("card_temporal_anisotropy", train_cfg.card_temporal_anisotropy, bool),
        ("card_temporal_anisotropy_q", train_cfg.card_temporal_anisotropy_q, float),
        ("card_temporal_operator_type", train_cfg.card_temporal_operator_type, str),
        (
            "card_temporal_operator_spectral_transform",
            train_cfg.card_temporal_operator_spectral_transform,
            str,
        ),
        ("card_temporal_velocity_weight", train_cfg.card_temporal_velocity_weight, float),
        (
            "card_temporal_acceleration_weight",
            train_cfg.card_temporal_acceleration_weight,
            float,
        ),
        (
            "card_temporal_jerk_weight",
            train_cfg.card_temporal_jerk_weight,
            float,
        ),
        (
            "card_temporal_anisotropy_learned_from_history",
            train_cfg.card_temporal_anisotropy_learned_from_history,
            bool,
        ),
        (
            "card_temporal_anisotropy_history_dim",
            train_cfg.card_temporal_anisotropy_history_dim,
            int,
        ),
        (
            "card_temporal_anisotropy_delta_max_abs",
            train_cfg.card_temporal_anisotropy_delta_max_abs,
            float,
        ),
        ("card_node_covariance_type", train_cfg.card_node_covariance_type, str),
        ("card_mobility_palm_var", train_cfg.card_mobility_palm_var, float),
        ("card_mobility_depth1_var", train_cfg.card_mobility_depth1_var, float),
        ("card_mobility_depth2_var", train_cfg.card_mobility_depth2_var, float),
        ("card_mobility_depth3plus_var", train_cfg.card_mobility_depth3plus_var, float),
        ("card_dhalf_gamma", train_cfg.card_dhalf_gamma, float),
        ("card_learnable_dhalf", train_cfg.card_learnable_dhalf, bool),
        ("card_graph_laplacian_alpha", train_cfg.card_graph_laplacian_alpha, float),
        ("card_graph_laplacian_beta", train_cfg.card_graph_laplacian_beta, float),
        ("card_graph_laplacian_normalized", train_cfg.card_graph_laplacian_normalized, bool),
        ("card_denoiser_dim", train_cfg.card_denoiser_dim, int),
        ("card_denoiser_depth", train_cfg.card_denoiser_depth, int),
        ("card_denoiser_heads", train_cfg.card_denoiser_heads, int),
        ("card_dropout", train_cfg.card_dropout, float),
        ("card_freeze_coarse", train_cfg.card_freeze_coarse, bool),
        ("card_diffusion_coarse_warmup_epochs", train_cfg.card_diffusion_coarse_warmup_epochs, int),
        ("card_cond_use_history", train_cfg.card_cond_use_history, bool),
        ("card_cond_use_coarse", train_cfg.card_cond_use_coarse, bool),
        ("card_allow_no_conditioning", train_cfg.card_allow_no_conditioning, bool),
        ("card_diffusion_only", train_cfg.card_diffusion_only, bool),
        ("card_use_mamp_condition", train_cfg.card_use_mamp_condition, bool),
        ("card_use_mamp_condition_coarse", train_cfg.card_use_mamp_condition_coarse, bool),
        ("card_mamp_checkpoint", train_cfg.card_mamp_checkpoint, str),
        ("card_mamp_config", train_cfg.card_mamp_config, str),
        ("card_mamp_repo_root", train_cfg.card_mamp_repo_root, str),
        ("card_mamp_mask_ratio", train_cfg.card_mamp_mask_ratio, float),
        ("card_mamp_motion_aware_tau", train_cfg.card_mamp_motion_aware_tau, float),
        ("card_mpjpe_weight", train_cfg.card_mpjpe_weight, float),
        ("card_coarse_target_lowpass_only", train_cfg.card_coarse_target_lowpass_only, bool),
        ("card_graph_laplacian_tau", train_cfg.card_graph_laplacian_tau, float),
        ("card_covariance_jitter", train_cfg.card_covariance_jitter, float),
    ]
    card_nullable_fields = {
        "card_beta_matrix_min_rate",
        "card_beta_matrix_max_rate",
    }
    for key, value, caster in card_fields:
        if value is not None or key in card_nullable_fields:
            config[key] = None if value is None else caster(value)

    cfg_dict = asdict(train_cfg)
    cfg_dict["dataset"] = asdict(ds)

    run = None
    if log_wandb:
        try:
            import wandb

            project = wandb_project or os.environ.get("WANDB_PROJECT") or "pose-forecasting"
            run_name = (
                f"{wandb_run_prefix}-{tag}" if wandb_run_prefix else tag
            )
            run = wandb.init(
                project=project,
                entity=wandb_entity,
                name=run_name,
                config={"train": cfg_dict, "runtime": config},
                reinit=True,
            )
        except Exception as exc:
            print(f"[wandb] init skipped: {exc}")
            run = None

    try:
        out = train(
            config,
            epochs=train_cfg.epochs,
            lr=train_cfg.lr,
            bone_loss_weight=train_cfg.bone_loss_weight,
            model=train_cfg.model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_wandb=bool(run is not None),
            wandb_run=run,
        )
    finally:
        if run is not None:
            try:
                wandb_payload = _prepare_wandb_metrics(out or {}) if isinstance(out, dict) else {}
                if wandb_payload:
                    run.log(wandb_payload)
            except Exception:
                pass
            run.finish()

    if out is None:
        return None
    out = dict(out)
    out["run_tag"] = tag
    out["elapsed_seconds"] = float(out.get("elapsed_seconds", 0.0))
    return out
