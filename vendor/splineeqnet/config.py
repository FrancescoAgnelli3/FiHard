import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


DEFAULT_MODELS = "FiHard"
DATASET_CHOICES: Tuple[str, ...] = ("assembly", "h2o", "bighands", "fpha")

MODEL_DEFAULT_CONFIGS: Dict[str, Dict[str, object]] = {
    "simlpe_dct": {
        "model": "simlpe_dct",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 128,
        "gru_layers": 8,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "dct_keep_coeffs": None,
        "simlpe_norm_axis": "all",
        "simlpe_use_norm": None,
        "simlpe_use_spatial_fc_only": None,
        "simlpe_mix_spatial_temporal": None,
        "simlpe_add_last_offset": True,
        "epochs": 50,
        "early_stopping_enabled": True,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-4,
        "early_stopping_warmup": 10,
        "early_stopping_monitor": "auto",
    },
    "fihard": {
        "model": "FiHard",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 128,
        "gru_layers": 8,
        "gradient_clip": 0,
        "bone_loss_weight": 0.0,
        "dct_keep_coeffs": None,
        "simlpe_norm_axis": "all",
        "simlpe_use_norm": None,
        "simlpe_use_spatial_fc_only": None,
        "simlpe_mix_spatial_temporal": None,
        "simlpe_add_last_offset": True,
        "FiHard_k_low": 80,
        "FiHard_diffusion_epochs": 500,
        "FiHard_diffusion_steps": 100,
        "FiHard_ddim_steps": 50,
        "FiHard_isotropic_noise": False,
        "FiHard_spatial_anisotropy": True,
        "FiHard_beta_matrix_power": 1.0,
        "FiHard_beta_matrix_min_rate": 0.5,
        "FiHard_beta_matrix_max_rate": 2.0,
        "FiHard_temporal_anisotropy": False,
        "FiHard_temporal_anisotropy_q": 1.0,
        "FiHard_temporal_operator_type": "identity",
        "FiHard_temporal_operator_spectral_transform": "identity_shifted_power",
        "FiHard_temporal_velocity_weight": 1.0,
        "FiHard_temporal_acceleration_weight": 1.0,
        "FiHard_temporal_jerk_weight": 1.0,
        "FiHard_temporal_anisotropy_learned_from_history": False,
        "FiHard_temporal_anisotropy_history_dim": 128,
        "FiHard_temporal_anisotropy_delta_max_abs": 2.0,
        "FiHard_node_covariance_type": "laplacian_heat_kernel",
        "FiHard_mobility_palm_var": 0.15,
        "FiHard_mobility_depth1_var": 0.35,
        "FiHard_mobility_depth2_var": 0.70,
        "FiHard_mobility_depth3plus_var": 1.00,
        "FiHard_dhalf_gamma": 1.0,
        "FiHard_learnable_dhalf": False,
        "FiHard_graph_laplacian_alpha": 0.0,
        "FiHard_graph_laplacian_beta": 1.0,
        "FiHard_graph_laplacian_normalized": True,
        "FiHard_denoiser_dim": 256,
        "FiHard_denoiser_depth": 6,
        "FiHard_denoiser_heads": 8,
        "FiHard_dropout": 0.0,
        "FiHard_freeze_coarse": True,
        "FiHard_diffusion_coarse_warmup_epochs": 10,
        "FiHard_cond_use_history": False,
        "FiHard_cond_use_coarse": True,
        "FiHard_allow_no_conditioning": False,
        "FiHard_diffusion_only": False,
        "FiHard_use_mamp_condition": False,
        "FiHard_use_mamp_condition_coarse": False,
        "FiHard_mamp_checkpoint": "",
        "FiHard_mamp_config": "",
        "FiHard_mamp_repo_root": "",
        "FiHard_mamp_mask_ratio": 0.0,
        "FiHard_mamp_motion_aware_tau": 0.80,
        "FiHard_mpjpe_weight": 0.0,
        "FiHard_coarse_target_lowpass_only": False,
        "epochs": 50,
        "early_stopping_enabled": True,
        "early_stopping_patience": 20,
        "early_stopping_min_delta": 1e-4,
        "early_stopping_warmup": 10,
        "early_stopping_monitor": "auto",
    },
}


@dataclass
class DatasetCfg:
    data_dir: str
    action_filter: str = "pick_up_screwd"
    input_n: int = 90
    output_n: int = 10
    stride: int = 5
    time_interp: Optional[int] = None
    window_norm: Optional[int] = None
    batch_size: int = 128
    eval_batch_mult: int = 1
    seed: int = 0
    wrist_indices: Tuple[int, ...] = ()
    dataset: str = "assembly"
    node_count: int = 21
    edge_index: Tuple[Tuple[int, int], ...] = tuple()
    adjacency: Tuple[Tuple[int, ...], ...] = tuple()
    subset_files: Optional[Tuple[str, ...]] = None


@dataclass
class TrainCfg:
    model: str = "FiHard"
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 128
    gru_layers: int = 8
    gradient_clip: float = 10.0
    bone_loss_weight: float = 0.0
    velocity_loss_weight: float = 0.0
    save_eval_examples: bool = False
    save_coarse_model: bool = False
    use_space: bool = True
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    early_stopping_warmup: int = 0
    early_stopping_monitor: str = "auto"

    # Shared SiMLPe-DCT knobs
    dct_keep_coeffs: Optional[int] = 100
    simlpe_norm_axis: Optional[str] = None
    simlpe_use_norm: Optional[bool] = None
    simlpe_use_spatial_fc_only: Optional[bool] = None
    simlpe_mix_spatial_temporal: Optional[bool] = None
    simlpe_add_last_offset: Optional[bool] = None

    # Twostage diffusion knobs
    FiHard_diffusion_epochs: Optional[int] = None
    FiHard_k_low: Optional[int] = None
    FiHard_diffusion_steps: Optional[int] = None
    FiHard_ddim_steps: Optional[int] = None
    FiHard_isotropic_noise: Optional[bool] = None
    FiHard_spatial_anisotropy: Optional[bool] = None
    FiHard_beta_matrix_power: Optional[float] = None
    FiHard_beta_matrix_min_rate: Optional[float] = None
    FiHard_beta_matrix_max_rate: Optional[float] = None
    FiHard_temporal_anisotropy: Optional[bool] = None
    FiHard_temporal_anisotropy_q: Optional[float] = None
    FiHard_temporal_operator_type: Optional[str] = None
    FiHard_temporal_operator_spectral_transform: Optional[str] = None
    FiHard_temporal_velocity_weight: Optional[float] = None
    FiHard_temporal_acceleration_weight: Optional[float] = None
    FiHard_temporal_jerk_weight: Optional[float] = None
    FiHard_temporal_anisotropy_learned_from_history: Optional[bool] = None
    FiHard_temporal_anisotropy_history_dim: Optional[int] = None
    FiHard_temporal_anisotropy_delta_max_abs: Optional[float] = None
    FiHard_node_covariance_type: Optional[str] = None
    FiHard_mobility_palm_var: Optional[float] = None
    FiHard_mobility_depth1_var: Optional[float] = None
    FiHard_mobility_depth2_var: Optional[float] = None
    FiHard_mobility_depth3plus_var: Optional[float] = None
    FiHard_dhalf_gamma: Optional[float] = None
    FiHard_learnable_dhalf: Optional[bool] = None
    FiHard_graph_laplacian_alpha: Optional[float] = None
    FiHard_graph_laplacian_beta: Optional[float] = None
    FiHard_graph_laplacian_normalized: Optional[bool] = None
    FiHard_denoiser_dim: Optional[int] = None
    FiHard_denoiser_depth: Optional[int] = None
    FiHard_denoiser_heads: Optional[int] = None
    FiHard_dropout: Optional[float] = None
    FiHard_freeze_coarse: Optional[bool] = None
    FiHard_diffusion_coarse_warmup_epochs: Optional[int] = None
    FiHard_cond_use_history: Optional[bool] = None
    FiHard_cond_use_coarse: Optional[bool] = None
    FiHard_allow_no_conditioning: Optional[bool] = None
    FiHard_diffusion_only: Optional[bool] = None
    FiHard_use_mamp_condition: Optional[bool] = None
    FiHard_use_mamp_condition_coarse: Optional[bool] = None
    FiHard_mamp_checkpoint: Optional[str] = None
    FiHard_mamp_config: Optional[str] = None
    FiHard_mamp_repo_root: Optional[str] = None
    FiHard_mamp_mask_ratio: Optional[float] = None
    FiHard_mamp_motion_aware_tau: Optional[float] = None
    FiHard_mpjpe_weight: Optional[float] = None
    FiHard_coarse_target_lowpass_only: Optional[bool] = None
    FiHard_graph_laplacian_tau: Optional[float] = None
    FiHard_covariance_jitter: Optional[float] = None


def parse_list(arg: str, cast):
    return [cast(x) for x in arg.split(',') if x != '']


def build_argument_parser(base_dir: str, default_data_dir: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run FiHard experiments.")
    ap.add_argument("--dataset", type=str, default="assembly", choices=DATASET_CHOICES)
    ap.add_argument("--data-dir", type=str, default=default_data_dir)
    ap.add_argument("--action-filter", type=str, default="")
    ap.add_argument("--input-n", type=int, default=70)
    ap.add_argument("--output-n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--time-interp", type=int, default=None)
    ap.add_argument("--window-norm", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-mult", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrist-indices", type=str, default="")
    ap.add_argument("--models", type=str, default=DEFAULT_MODELS)
    ap.add_argument("--hidden-sizes", type=str, default="128")
    ap.add_argument("--gru-layers", type=str, default="8")
    ap.add_argument("--lrs", type=str, default="0.001")
    ap.add_argument("--bone-loss-weights", type=str, default="0.0")
    ap.add_argument("--gradient-clips", type=str, default="0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--save-root", type=str, default=os.path.join(base_dir, "out", "ablation"))
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--save-eval-examples", action="store_true")
    return ap
