import copy
import math
import os
import sys
import time
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["train"]

# Keep hard dependencies minimal for card-only runtime.
from models.simlpe_dct import SiMLPeDCTConfig, SiMLPeDCTForecaster
from models.card import CardConfig, CardForecaster

import utils as U

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from common.evaluation import (
    humanmac_metrics_prefixed as _common_humanmac_metrics_prefixed,  # type: ignore
    save_eval_samples_npz,
)

# Minimal hand graph helpers kept for card runtime compatibility.
_HAND_BONE_LINK_1 = []
_HAND_BONE_LINK_2 = []


def _HAND_BUILD_UNDIRECTED(node_num: int, links):
    graph = {i: [] for i in range(node_num)}
    for a, b in links:
        if 0 <= a < node_num and 0 <= b < node_num:
            graph[a].append(b)
            graph[b].append(a)
    return graph


def _HAND_BUILD_TREE(node_num: int, links, wrists_hint=None):
    wrists = list(wrists_hint or [])
    return None, None, wrists, None


def _compute_humanmac_metrics(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    conditioning_context: torch.Tensor,
    *,
    threshold: float,
) -> Dict[str, float]:
    return _common_humanmac_metrics_prefixed(
        pred_candidates=pred_candidates,
        gt_future=gt_future,
        conditioning_context=conditioning_context,
        threshold=threshold,
    )


def _enable_runtime_acceleration() -> None:
    tf32_env = os.environ.get("DIFFHF_ENABLE_TF32", "1").strip().lower()
    sdp_env = os.environ.get("DIFFHF_ENABLE_SDP", "1").strip().lower()
    use_tf32 = tf32_env not in {"0", "false", "no", "off"}
    use_sdp = sdp_env not in {"0", "false", "no", "off"}
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = use_tf32
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = use_tf32
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_flash_sdp(use_sdp)
            torch.backends.cuda.enable_mem_efficient_sdp(use_sdp)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def _maybe_compile_module(module: Any, *, name: str):
    compile_env = os.environ.get("DIFFHF_ENABLE_COMPILE", "1").strip().lower()
    if compile_env in {"0", "false", "no", "off"}:
        return module
    scoped_env_key = f"DIFFHF_ENABLE_COMPILE_{name.upper().replace('.', '_')}"
    scoped_default = "0" if name == "card.coarse" else "1"
    scoped_env = os.environ.get(scoped_env_key, scoped_default).strip().lower()
    if scoped_env in {"0", "false", "no", "off"}:
        return module
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None or module is None:
        return module
    try:
        compiled = compile_fn(module, mode="reduce-overhead")
        print(f"[Info] torch.compile enabled for {name}")
        return compiled
    except Exception as exc:
        print(f"[Warn] torch.compile skipped for {name}: {exc}")
        return module

def _print_model_parameters(model_or_modules, title: str = "Model") -> int:
    """Calculate and print the number of trainable parameters.

    Accepts a single `nn.Module` or an iterable of modules. Returns the total
    count of trainable parameters.
    """
    def count_params(m) -> int:
        total = 0
        for p in m.parameters():
            if not getattr(p, "requires_grad", False):
                continue
            try:
                total += int(p.numel())
            except Exception:
                # Skip uninitialized lazy parameters; they initialize on first forward pass.
                continue
        return total

    if isinstance(model_or_modules, (list, tuple)):
        total = sum(count_params(m) for m in model_or_modules)
    else:
        total = count_params(model_or_modules)

    # Human-friendly formatting
    if total >= 1_000_000:
        pretty = f"{total/1_000_000:.2f}M"
    elif total >= 1_000:
        pretty = f"{total/1_000:.2f}K"
    else:
        pretty = str(total)

    print(f"{title} parameters: {total} ({pretty})")
    return total


def _estimate_empirical_temporal_mean_covariance(
    loader: DataLoader,
    *,
    wrist_index: int,
    node_num: int,
) -> torch.Tensor:
    free_indices = [i for i in range(int(node_num)) if i != int(wrist_index)]
    if not free_indices:
        return torch.zeros((0, 0), dtype=torch.float32)

    batch_means: List[torch.Tensor] = []
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Loader batch must provide input and target tensors.")
        out = batch[1]
        if out.ndim != 4 or out.size(-1) < 7:
            raise ValueError(
                "Expected target tensor with shape (B, T, N, C>=7) so target 3D coordinates can be read from out[..., 4:]."
            )
        tgt_3d = out[:, :, :, 4:].to(dtype=torch.float64, device="cpu")
        if tgt_3d.shape[2] != node_num or tgt_3d.shape[3] != 3:
            raise ValueError(
                f"Expected target 3D tensor shape (B, T, {node_num}, 3), got {tuple(tgt_3d.shape)}"
            )
        pose_mean = tgt_3d.mean(dim=1)
        pose_free = pose_mean[:, free_indices, :].reshape(pose_mean.size(0), -1)
        batch_means.append(pose_free)

    if not batch_means:
        raise ValueError("Cannot estimate empirical covariance from an empty training loader.")

    samples = torch.cat(batch_means, dim=0)
    if samples.size(0) < 2:
        raise ValueError("At least two training samples are required to estimate empirical covariance.")

    centered = samples - samples.mean(dim=0, keepdim=True)
    cov = centered.t().matmul(centered) / float(samples.size(0) - 1)
    return cov.to(dtype=torch.float32)

def train(
    config: dict,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    bone_loss_weight: float = 0.0,
    model: str = "card",
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    log_wandb: bool = False,
    wandb_run: Optional[Any] = None,
):
    _enable_runtime_acceleration()
    device = U.device()

    raw_model_name = str(model or "").strip()
    model_key = raw_model_name.lower()
    if not model_key:
        model_key = "card"
    model = model_key

    log_wandb = bool(log_wandb or config.get("log_wandb", False))
    wandb_handle = wandb_run
    if log_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            print("Weights & Biases requested for logging but package is unavailable; disabling logging.")
            log_wandb = False
        else:
            if wandb_handle is None:
                wandb_handle = wandb.run
            if wandb_handle is None:
                print("Weights & Biases logging requested without an active run; call wandb.init() before train(). Disabling logging.")
                log_wandb = False

    def _maybe_add_metric(bucket: Dict[str, float], key: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(scalar):
            bucket[key] = scalar

    save_eval_examples = bool(config.get("save_eval_examples", False))
    save_coarse_model = bool(config.get("save_coarse_model", False))
    maybe_save_coarse_model = None
    params_count = 0
    node_num = int(config.get("node_num", 21))
    input_n = int(config.get("input_n", 90))
    output_n = int(config.get("output_n", 42))
    hidden_size = int(config.get("hidden_size", 128))
    t_layers = int(config.get("t_layers", 2))
    gnn_layers = int(config.get("gnn_layers", 2))
    batch_size = int(config.get("batch_size", 16))
    space = bool(config.get("use_space", True))
    gru_layers = int(config.get("gru_layers", 4))
    train_epochs = int(epochs if epochs is not None else config.get("train_epoches", 50))
    learning_rate = float(lr if lr is not None else config.get("learning_rate", 1e-3))
    card_diffusion_epochs = int(config.get("card_diffusion_epochs", 0))
    num_candidates = max(
        1,
        int(
            config.get(
                "num_candidates",
                config.get("card_eval_best_of_k", config.get("humanmac_num_candidates", 1)),
            )
        ),
    )
    card_eval_best_of_k = num_candidates
    card_mamp_checkpoint = str(config.get("card_mamp_checkpoint", "")).strip()
    card_use_mamp_condition = bool(config.get("card_use_mamp_condition", bool(card_mamp_checkpoint)))
    card_use_mamp_condition_coarse = bool(config.get("card_use_mamp_condition_coarse", False))
    card_use_any_mamp_condition = bool(card_use_mamp_condition or card_use_mamp_condition_coarse)
    card_mamp_mask_ratio = float(config.get("card_mamp_mask_ratio", 0.0))
    card_mamp_motion_aware_tau = float(config.get("card_mamp_motion_aware_tau", 0.80))
    card_mamp_repo_root = str(config.get("card_mamp_repo_root", os.path.join(_PROJECT_ROOT, "vendor", "mamp"))).strip()
    dct_keep_coeffs_cfg = config.get("dct_keep_coeffs")
    dct_keep_coeffs = int(dct_keep_coeffs_cfg) if dct_keep_coeffs_cfg is not None else None
    velocity_loss_weight = float(config.get("velocity_loss_weight", 0.0))

    adjacency_cfg = config.get("adjacency", ())
    adjacency_tensor = None
    if adjacency_cfg:
        adjacency_tensor = torch.tensor(adjacency_cfg, dtype=torch.float32)
    else:
        adjacency_tensor = torch.eye(node_num, dtype=torch.float32)
    if adjacency_tensor.ndim != 2:
        raise ValueError("Adjacency matrix must be a 2D square array.")
    if adjacency_tensor.shape[0] != node_num or adjacency_tensor.shape[1] != node_num:
        adj_resized = torch.zeros((node_num, node_num), dtype=torch.float32)
        rows = min(node_num, adjacency_tensor.shape[0])
        cols = min(node_num, adjacency_tensor.shape[1])
        adj_resized[:rows, :cols] = adjacency_tensor[:rows, :cols]
        adjacency_tensor = adj_resized
    adjacency_tensor = adjacency_tensor.clone().detach()
    adjacency_tensor = 0.5 * (adjacency_tensor + adjacency_tensor.transpose(0, 1))
    diag_indices = torch.arange(node_num)
    adjacency_tensor[diag_indices, diag_indices] = 1.0

    edge_index_cfg = config.get("edge_index", ())
    edge_pairs: List[Tuple[int, int]] = []
    for pair in edge_index_cfg:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if 0 <= i < node_num and 0 <= j < node_num:
            edge_pairs.append((i, j))
    if not edge_pairs:
        nz = adjacency_tensor.nonzero(as_tuple=False)
        edge_pairs = [(int(i), int(j)) for i, j in nz]

    wrists_hint_cfg = config.get("wrists_hint")
    wrists_hint: Optional[List[int]] = None
    if wrists_hint_cfg is not None:
        if isinstance(wrists_hint_cfg, str):
            raw_tokens = wrists_hint_cfg.replace(";", ",").split(",")
            parsed: List[int] = []
            for token in raw_tokens:
                token = token.strip()
                if not token:
                    continue
                try:
                    parsed.append(int(float(token)))
                except ValueError:
                    continue
            filtered = [w for w in parsed if 0 <= w < node_num]
            wrists_hint = filtered if filtered else None
        elif isinstance(wrists_hint_cfg, (list, tuple)):
            parsed: List[int] = []
            for token in wrists_hint_cfg:
                try:
                    parsed.append(int(float(token)))
                except (ValueError, TypeError):
                    continue
            filtered = [w for w in parsed if 0 <= w < node_num]
            wrists_hint = filtered if filtered else None

    wrist_links = _HAND_BONE_LINK_1 + _HAND_BONE_LINK_2
    try:
        wrist_graph = _HAND_BUILD_UNDIRECTED(node_num, wrist_links)
    except Exception:
        wrist_graph = {i: [] for i in range(node_num)}
    try:
        _, _, wrists_auto, _ = _HAND_BUILD_TREE(node_num, wrist_links, wrists_hint)
    except Exception:
        wrists_auto = []
    wrists_candidates: List[int] = []
    if wrists_hint:
        wrists_candidates.extend(wrists_hint)
    wrists_candidates.extend(int(w) for w in wrists_auto if 0 <= int(w) < node_num)

    seen_wrists = set()
    ordered_wrists: List[int] = []
    for wrist_id in wrists_candidates:
        if wrist_id in seen_wrists:
            continue
        seen_wrists.add(wrist_id)
        ordered_wrists.append(wrist_id)

    fallback_wrists = [idx for idx in (5, 26) if idx < node_num]
    if not ordered_wrists:
        ordered_wrists = fallback_wrists or [0]

    node2wrist_map: List[int] = [-1] * node_num
    for root in ordered_wrists:
        stack = [root]
        while stack:
            node = stack.pop()
            if node2wrist_map[node] != -1:
                continue
            node2wrist_map[node] = root
            for neigh in wrist_graph.get(node, []):
                if 0 <= neigh < node_num and node2wrist_map[neigh] == -1:
                    stack.append(neigh)
    for idx, owner in enumerate(node2wrist_map):
        if owner == -1:
            node2wrist_map[idx] = ordered_wrists[0]
    node2wrist_tensor = torch.tensor(node2wrist_map, dtype=torch.long, device=device)

    def _delta_to_absolute(pred: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
        pred = pred.clone()
        pred[:, 0, :] = pred[:, 0, :] + last_frame
        return torch.cumsum(pred, dim=1)

    def _velocity_magnitude(future_seq: torch.Tensor, last_observed: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep velocity magnitudes for future sequence.

        Args:
            future_seq: Tensor shaped (B, T_out, N, 3) with predicted positions.
            last_observed: Tensor shaped (B, N, 3) representing final observed pose.

        Returns:
            Tensor shaped (B, T_out, N) containing velocity magnitudes.
        """
        if future_seq.ndim != 4 or last_observed.ndim != 3:
            raise ValueError("Invalid shapes for velocity magnitude computation.")
        prev = torch.cat([last_observed.unsqueeze(1), future_seq[:, :-1]], dim=1)
        diffs = future_seq - prev
        return torch.norm(diffs, dim=-1)

    loader = train_loader

    if edge_pairs:
        edge_index_tensor_device = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index_tensor_device = torch.zeros((2, 0), dtype=torch.long, device=device)
    graph_edges_no_self = [(i, j) for (i, j) in edge_pairs if i != j]
    if graph_edges_no_self:
        edges_t = torch.tensor(graph_edges_no_self, dtype=torch.long, device=device)
    else:
        edges_t = torch.zeros((0, 2), dtype=torch.long, device=device)

    # Align model batch_size with the actual train loader to avoid GRU hidden-state mismatches
    if hasattr(loader, "batch_size") and loader.batch_size is not None:
        try:
            batch_size = int(loader.batch_size)
        except Exception:
            pass

    # Build models and optimizer
    params: List[torch.nn.Parameter] = []
    card_model = None
    card_phase: Optional[str] = None
    card_diffusion_lr = float(learning_rate) * 0.1
    card_train_coarse_in_diffusion = False
    card_diffusion_coarse_warmup_epochs = 0
    card_coarse_diffusion_group_added = False
    mamp_encoder = None
    mamp_num_frames = input_n
    contrastive_model = None
    contrastive_path = str(config.get("contrastive_score_path", "") or "")
    contrastive_use = bool(config.get("contrastive_score_use", True))

    def _load_contrastive_model():
        nonlocal contrastive_model
        if not contrastive_use:
            return None
        if contrastive_model is not None:
            return contrastive_model

        dataset_name = str(config.get("dataset", "") or "").strip()
        action_segment = str(config.get("action_filter", "") or "all")
        action_segment = action_segment.replace(os.sep, "_").replace(" ", "_")
        default_name = f"contrastive_score_{dataset_name}_{action_segment}.pt" if dataset_name else "contrastive_score.pt"
        default_path = os.path.join(os.path.dirname(__file__), "examples", "models", default_name)
        load_path = contrastive_path or default_path

        if not os.path.exists(load_path):
            raise RuntimeError(
                "Contrastive scorer checkpoint not found. "
                f"Expected at {load_path}. "
                "Provide it via config['contrastive_score_path'] or train the model with train_contrastive_score.py."
            )

        checkpoint = torch.load(load_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            dataset_cfg = checkpoint.get("dataset_cfg", {}) or {}
            n_joints = int(dataset_cfg.get("node_count", 21))
        else:
            state = checkpoint
            n_joints = int(config.get("node_num", 21))

        embed_dim = None
        for key in ("x_enc.proj.3.weight", "x_enc.proj.2.weight", "x_enc.proj.1.weight"):
            weight = state.get(key)
            if weight is not None and hasattr(weight, "shape"):
                embed_dim = int(weight.shape[0])
                break
        if embed_dim is None:
            raise RuntimeError("Unable to infer contrastive embed_dim from checkpoint state dict.")

        from models.contrastive.contrastive_score import ContrastiveScorer

        contrastive_model = ContrastiveScorer(n_joints=n_joints, embed_dim=embed_dim, dropout=0.0, share_encoders=False)
        missing, unexpected = contrastive_model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Warn] Contrastive scorer missing keys: {missing}")
        if unexpected:
            print(f"[Warn] Contrastive scorer unexpected keys: {unexpected}")
        contrastive_model = contrastive_model.to(device)
        contrastive_model.eval()
        return contrastive_model
    def _parse_fixed_centers_cfg(raw_value: Any) -> Optional[Tuple[float, float]]:
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            tokens = [tok.strip() for tok in raw_value.replace("|", ",").split(",") if tok.strip()]
            if len(tokens) >= 2:
                return float(tokens[0]), float(tokens[1])
        elif isinstance(raw_value, (list, tuple)):
            if len(raw_value) >= 2:
                return float(raw_value[0]), float(raw_value[1])
        return None

    if model == "simlpe_dct":
        sim_cfg = SiMLPeDCTConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
            dct_components=dct_keep_coeffs,
        )
        simlpe_dct = SiMLPeDCTForecaster(sim_cfg).to(device)
        simlpe_dct = _maybe_compile_module(simlpe_dct, name="simlpe_dct")
        params = list(simlpe_dct.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe_dct, title="SiMLPeDCTForecaster")
    elif model == "card":
        # Two-stage: coarse low-band DCT predictor + conditional diffusion over future-DCT residuals.
        k_low = int(config.get("card_k_low", 16))
        diff_steps = int(config.get("card_diffusion_steps", 100))
        ddim_steps = int(config.get("card_ddim_steps", 50))
        isotropic_noise = bool(config.get("card_isotropic_noise", False))
        spatial_anisotropy = bool(config.get("card_spatial_anisotropy", True))
        beta_matrix_power = float(config.get("card_beta_matrix_power", 1.0))
        beta_matrix_min_rate_raw = config.get("card_beta_matrix_min_rate", 0.5)
        beta_matrix_max_rate_raw = config.get("card_beta_matrix_max_rate", 2.0)
        beta_matrix_min_rate = None if beta_matrix_min_rate_raw is None else float(beta_matrix_min_rate_raw)
        beta_matrix_max_rate = None if beta_matrix_max_rate_raw is None else float(beta_matrix_max_rate_raw)
        temporal_anisotropy = bool(config.get("card_temporal_anisotropy", False))
        temporal_anisotropy_q = float(config.get("card_temporal_anisotropy_q", 1.0))
        temporal_operator_type = str(config.get("card_temporal_operator_type", "identity"))
        temporal_operator_spectral_transform = str(
            config.get("card_temporal_operator_spectral_transform", "identity_shifted_power")
        )
        temporal_velocity_weight = float(config.get("card_temporal_velocity_weight", 1.0))
        temporal_acceleration_weight = float(config.get("card_temporal_acceleration_weight", 1.0))
        temporal_jerk_weight = float(config.get("card_temporal_jerk_weight", 1.0))
        temporal_anisotropy_learned_from_history = bool(
            config.get("card_temporal_anisotropy_learned_from_history", False)
        )
        temporal_anisotropy_history_dim = int(
            config.get("card_temporal_anisotropy_history_dim", 128)
        )
        temporal_anisotropy_delta_max_abs = float(
            config.get("card_temporal_anisotropy_delta_max_abs", 2.0)
        )
        node_covariance_type = str(config.get("card_node_covariance_type", "laplacian_heat_kernel"))
        mobility_palm_var = float(config.get("card_mobility_palm_var", 0.15))
        mobility_depth1_var = float(config.get("card_mobility_depth1_var", 0.35))
        mobility_depth2_var = float(config.get("card_mobility_depth2_var", 0.70))
        mobility_depth3plus_var = float(config.get("card_mobility_depth3plus_var", 1.00))
        dhalf_gamma = float(config.get("card_dhalf_gamma", 1.0))
        learnable_dhalf = bool(config.get("card_learnable_dhalf", False))
        graph_laplacian_alpha = float(config.get("card_graph_laplacian_alpha", 0.0))
        graph_laplacian_beta = float(config.get("card_graph_laplacian_beta", 1.0))
        graph_laplacian_normalized = bool(config.get("card_graph_laplacian_normalized", True))
        d_model = int(config.get("card_denoiser_dim", 256))
        depth = int(config.get("card_denoiser_depth", 6))
        n_heads = int(config.get("card_denoiser_heads", 8))
        p_drop = float(config.get("card_dropout", 0.0))
        freeze_coarse = bool(config.get("card_freeze_coarse", True))
        # If coarse is frozen, keep it frozen during diffusion.
        # Only allow diffusion-stage coarse updates when freeze_coarse is false.
        card_train_coarse_in_diffusion = not bool(freeze_coarse)
        card_diffusion_coarse_warmup_epochs = max(
            0, int(config.get("card_diffusion_coarse_warmup_epochs", 10))
        )
        card_diffusion_lr = float(learning_rate) * 0.1
        cond_use_history = bool(config.get("card_cond_use_history", True))
        cond_use_coarse = bool(config.get("card_cond_use_coarse", True))
        allow_no_conditioning = bool(config.get("card_allow_no_conditioning", False))
        coarse_target_lowpass_only = bool(config.get("card_coarse_target_lowpass_only", False))
        diffusion_only = bool(config.get("card_diffusion_only", False))
        graph_laplacian_tau = float(config.get("card_graph_laplacian_tau", 1.0))
        covariance_jitter = float(config.get("card_covariance_jitter", 1e-4))
        card_use_mamp_condition = bool(config.get("card_use_mamp_condition", False))
        card_use_mamp_condition_coarse = bool(config.get("card_use_mamp_condition_coarse", False))
        card_use_any_mamp_condition = (
            card_use_mamp_condition or card_use_mamp_condition_coarse
        )
        if diffusion_only:
            cond_use_coarse = False
            if not cond_use_history and not card_use_any_mamp_condition:
                allow_no_conditioning = True
            card_train_coarse_in_diffusion = False
        if not (
            cond_use_history
            or cond_use_coarse
            or card_use_any_mamp_condition
            or allow_no_conditioning
        ):
            raise ValueError(
                "card has no conditioning source enabled. "
                "Set card_allow_no_conditioning=true to run a true unconditional mode."
            )
        card_wrist_cfg = config.get("card_wrist_index")
        if card_wrist_cfg is None:
            card_wrist_index = int(ordered_wrists[0]) if ordered_wrists else 0
        else:
            card_wrist_index = int(card_wrist_cfg)
        if not (0 <= card_wrist_index < node_num):
            raise ValueError(
                f"card_wrist_index must be in [0, {node_num - 1}], got {card_wrist_index}"
            )

        raw_card_links = config.get("card_links", ())
        card_links: List[Tuple[int, int]] = []
        if isinstance(raw_card_links, (list, tuple)):
            for pair in raw_card_links:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    i, j = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                if 0 <= i < node_num and 0 <= j < node_num and i != j:
                    if i > j:
                        i, j = j, i
                    card_links.append((i, j))
        if not card_links:
            card_links = []
            for i, j in edge_pairs:
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                card_links.append((int(i), int(j)))
        card_links = sorted(set(card_links))
        if not card_links and not (isotropic_noise or not spatial_anisotropy):
            raise ValueError(
                "card requires non-empty hand links. "
                "Provide config['card_links'], ensure edge_index/adjacency includes hand edges, "
                "enable card_isotropic_noise, or set card_spatial_anisotropy=false."
            )

        empirical_feature_covariance = None
        if spatial_anisotropy and not isotropic_noise and node_covariance_type in {
            "empirical_temporal_mean",
            "empirical_heat_kernel",
            "empirical_plus_theoretical",
        }:
            empirical_feature_covariance = _estimate_empirical_temporal_mean_covariance(
                loader,
                wrist_index=int(card_wrist_index),
                node_num=node_num,
            )

        tw_cfg = CardConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            k_low=k_low,
            diffusion_steps=diff_steps,
            ddim_steps=ddim_steps,
            isotropic_noise=isotropic_noise,
            spatial_anisotropy=spatial_anisotropy,
            beta_matrix_power=beta_matrix_power,
            beta_matrix_min_rate=beta_matrix_min_rate,
            beta_matrix_max_rate=beta_matrix_max_rate,
            temporal_anisotropy=temporal_anisotropy,
            temporal_anisotropy_q=temporal_anisotropy_q,
            temporal_operator_type=temporal_operator_type,
            temporal_operator_spectral_transform=temporal_operator_spectral_transform,
            temporal_velocity_weight=temporal_velocity_weight,
            temporal_acceleration_weight=temporal_acceleration_weight,
            temporal_jerk_weight=temporal_jerk_weight,
            temporal_anisotropy_learned_from_history=temporal_anisotropy_learned_from_history,
            temporal_anisotropy_history_dim=temporal_anisotropy_history_dim,
            temporal_anisotropy_delta_max_abs=temporal_anisotropy_delta_max_abs,
            node_covariance_type=node_covariance_type,
            mobility_palm_var=mobility_palm_var,
            mobility_depth1_var=mobility_depth1_var,
            mobility_depth2_var=mobility_depth2_var,
            mobility_depth3plus_var=mobility_depth3plus_var,
            dhalf_gamma=dhalf_gamma,
            learnable_dhalf=learnable_dhalf,
            graph_laplacian_alpha=graph_laplacian_alpha,
            graph_laplacian_beta=graph_laplacian_beta,
            graph_laplacian_normalized=graph_laplacian_normalized,
            denoiser_dim=d_model,
            denoiser_depth=depth,
            denoiser_heads=n_heads,
            dropout=p_drop,
            freeze_coarse=freeze_coarse,
            cond_use_history=cond_use_history,
            cond_use_coarse=cond_use_coarse,
            allow_no_conditioning=allow_no_conditioning,
            coarse_target_lowpass_only=coarse_target_lowpass_only,
            diffusion_only=diffusion_only,
            graph_laplacian_tau=graph_laplacian_tau,
            covariance_jitter=covariance_jitter,
            simlpe_use_norm=bool(config.get("simlpe_use_norm", True)),
            simlpe_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            simlpe_mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            simlpe_norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            simlpe_add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
        )
        card_model = CardForecaster(
            tw_cfg,
            metadata={
                "wrist_index": int(card_wrist_index),
                "edges": tuple((int(i), int(j)) for i, j in card_links),
                "empirical_feature_covariance": empirical_feature_covariance,
            },
        ).to(device)
        card_model.coarse = _maybe_compile_module(card_model.coarse, name="card.coarse")
        card_model.diffusion.denoiser = _maybe_compile_module(
            card_model.diffusion.denoiser,
            name="card.diffusion.denoiser",
        )
        if card_use_any_mamp_condition:
            if not card_mamp_checkpoint:
                raise ValueError(
                    "MAMP conditioning requires card_mamp_checkpoint in config."
                )
            if card_mamp_repo_root:
                mamp_repo_abs = os.path.abspath(card_mamp_repo_root)
                if mamp_repo_abs not in sys.path:
                    sys.path.insert(0, mamp_repo_abs)
            try:
                from model_mamp.transformer import Transformer as MAMPTransformer  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"Failed to import MAMP Transformer: {exc}") from exc

            mamp_model_args: Optional[Dict[str, Any]] = None
            card_mamp_config = str(config.get("card_mamp_config", "")).strip()
            if card_mamp_config:
                try:
                    import yaml  # type: ignore

                    with open(card_mamp_config, "r", encoding="utf-8") as f:
                        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
                    if isinstance(yaml_cfg, dict) and isinstance(yaml_cfg.get("model_args"), dict):
                        mamp_model_args = dict(yaml_cfg["model_args"])
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to parse card_mamp_config={card_mamp_config}: {exc}"
                    ) from exc

            if mamp_model_args is None:
                mamp_model_args = {
                    "dim_in": 3,
                    "dim_feat": int(config.get("card_mamp_dim_feat", 256)),
                    "decoder_dim_feat": int(config.get("card_mamp_decoder_dim_feat", 256)),
                    "depth": int(config.get("card_mamp_depth", 8)),
                    "decoder_depth": int(config.get("card_mamp_decoder_depth", 5)),
                    "num_heads": int(config.get("card_mamp_num_heads", 8)),
                    "mlp_ratio": float(config.get("card_mamp_mlp_ratio", 4.0)),
                    "num_frames": int(config.get("card_mamp_num_frames", input_n)),
                    "num_joints": int(config.get("card_mamp_num_joints", node_num)),
                    "patch_size": int(config.get("card_mamp_patch_size", 1)),
                    "t_patch_size": int(config.get("card_mamp_t_patch_size", 1)),
                    "qkv_bias": True,
                    "qk_scale": None,
                    "drop_rate": 0.0,
                    "attn_drop_rate": 0.0,
                    "drop_path_rate": 0.0,
                    "norm_skes_loss": True,
                }
            mamp_num_frames = int(mamp_model_args.get("num_frames", input_n))
            if card_use_mamp_condition and mamp_num_frames != input_n:
                raise ValueError(
                    f"MAMP num_frames ({mamp_model_args.get('num_frames')}) must match input_n ({input_n}) "
                    "when card_use_mamp_condition=True."
                )
            if int(mamp_model_args.get("num_joints", node_num)) != node_num:
                raise ValueError(
                    f"MAMP num_joints ({mamp_model_args.get('num_joints')}) must match node_num ({node_num})."
                )

            mamp_encoder = MAMPTransformer(**mamp_model_args).to(device)
            try:
                ckpt = torch.load(card_mamp_checkpoint, map_location=device, weights_only=True)
            except Exception:
                ckpt = torch.load(card_mamp_checkpoint, map_location=device, weights_only=False)
            mamp_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = mamp_encoder.load_state_dict(mamp_state, strict=False)
            if missing:
                print(f"[Warning] MAMP checkpoint missing keys: {len(missing)}")
            if unexpected:
                print(f"[Warning] MAMP checkpoint unexpected keys: {len(unexpected)}")
            for p in mamp_encoder.parameters():
                p.requires_grad = False
            mamp_encoder.eval()
            print(f"[Info] Loaded frozen MAMP encoder from {card_mamp_checkpoint}")
        coarse_model_path = str(config.get("coarse_model_path", "")).strip() if save_coarse_model else ""
        coarse_model_saved = False

        def _maybe_save_coarse_model() -> None:
            nonlocal coarse_model_saved
            if not coarse_model_path or coarse_model_saved:
                return
            if os.path.exists(coarse_model_path):
                coarse_model_saved = True
                return
            os.makedirs(os.path.dirname(coarse_model_path), exist_ok=True)
            torch.save(
                {
                    "coarse_state": card_model.coarse.state_dict(),
                    "metadata": {
                        "tag": config.get("coarse_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                    },
                },
                coarse_model_path,
            )
            coarse_model_saved = True
            print(f"[Info] Saved coarse model to {coarse_model_path}")

        maybe_save_coarse_model = _maybe_save_coarse_model

        if coarse_model_path and os.path.exists(coarse_model_path):
            try:
                payload = torch.load(coarse_model_path, map_location=device, weights_only=False)
                if isinstance(payload, dict) and "coarse_state" in payload:
                    state = payload["coarse_state"]
                else:
                    state = payload
                missing, unexpected = card_model.coarse.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[Warning] Coarse model load had missing={len(missing)} unexpected={len(unexpected)} keys.")
                coarse_model_saved = True
                print(f"[Info] Loaded coarse model from {coarse_model_path}")
            except Exception as exc:
                print(f"[Warning] Failed to load coarse model from {coarse_model_path}: {exc}")

        def _set_card_phase(phase: str) -> List[torch.nn.Parameter]:
            nonlocal card_phase
            if phase == "coarse":
                for p in card_model.coarse.parameters():
                    p.requires_grad = True
                for p in card_model.diffusion.parameters():
                    p.requires_grad = False
                card_phase = "coarse"
                return [p for p in card_model.coarse.parameters() if p.requires_grad]
            if phase == "diffusion":
                for p in card_model.coarse.parameters():
                    p.requires_grad = False
                for p in card_model.diffusion.parameters():
                    p.requires_grad = True
                card_phase = "diffusion"
                return [p for p in card_model.diffusion.parameters() if p.requires_grad]
            raise ValueError(f"Unknown card phase: {phase}")

        params = _set_card_phase("coarse")
        eval_phase = str(config.get("card_eval_phase", "")).strip().lower()
        if eval_phase in {"coarse", "diffusion"} and train_epochs <= 0:
            params = _set_card_phase(eval_phase)
        init_lr = card_diffusion_lr if card_phase == "diffusion" else learning_rate
        optimizer = torch.optim.Adam(params, lr=init_lr)
        params_count = _print_model_parameters(card_model, title="CardForecaster")

    model_modules: Dict[str, torch.nn.Module] = {}
    if model == "simlpe_dct" and 'simlpe_dct' in locals():
        model_modules = {"simlpe_dct": simlpe_dct}
    elif model == "card" and card_model is not None:
        model_modules = {"card": card_model}

    def _set_mode(training: bool) -> None:
        for module in model_modules.values():
            if training:
                module.train()
            else:
                module.eval()

    def _capture_state() -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        if not model_modules:
            return None
        return {name: copy.deepcopy(module.state_dict()) for name, module in model_modules.items()}

    def _load_state(state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> None:
        if not state:
            return
        for name, module in model_modules.items():
            if name in state:
                module.load_state_dict(state[name])

    def _resize_mamp_time(coords_3d: torch.Tensor) -> torch.Tensor:
        if int(coords_3d.shape[1]) == int(mamp_num_frames):
            return coords_3d
        B, T, N, C = coords_3d.shape
        if C != 3:
            raise ValueError(f"Expected 3 coordinate channels, got {C}.")
        flat = coords_3d.reshape(B, T, N * C).transpose(1, 2).contiguous()  # (B, F, T)
        flat = F.interpolate(flat, size=int(mamp_num_frames), mode="linear", align_corners=False)
        return flat.transpose(1, 2).reshape(B, int(mamp_num_frames), N, C).contiguous()

    def _compute_mamp_feat(coords_3d: torch.Tensor) -> Optional[torch.Tensor]:
        if mamp_encoder is None:
            return None
        coords_3d = _resize_mamp_time(coords_3d)
        with torch.no_grad():
            latent, _mask, _ids_restore = mamp_encoder.forward_encoder(
                coords_3d,
                mask_ratio=card_mamp_mask_ratio,
                motion_aware_tau=card_mamp_motion_aware_tau,
            )
        return latent.mean(dim=1).detach()

    def _merge_mamp_features(
        hist_feat: Optional[torch.Tensor],
        coarse_feat: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if hist_feat is None:
            return coarse_feat
        if coarse_feat is None:
            return hist_feat
        if int(hist_feat.shape[0]) != int(coarse_feat.shape[0]):
            raise ValueError(
                f"MAMP batch mismatch while merging features: {hist_feat.shape[0]} vs {coarse_feat.shape[0]}"
            )
        return torch.stack([hist_feat, coarse_feat], dim=1).contiguous()

    def _set_dataset_aux_feature(ds: Any, feature_key: str, values: torch.Tensor) -> None:
        values_cpu = values.detach().cpu().float().contiguous()
        existing = getattr(ds, "aux_features", None)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged[feature_key] = values_cpu
            setattr(ds, "aux_features", merged)
            return
        if feature_key == "history":
            if hasattr(ds, "set_aux_features"):
                ds.set_aux_features(values_cpu)
            else:
                setattr(ds, "aux_features", values_cpu)
            return
        merged = {}
        if torch.is_tensor(existing):
            merged["history"] = existing.detach().cpu().float().contiguous()
        merged[feature_key] = values_cpu
        setattr(ds, "aux_features", merged)

    def _precompute_mamp_features_for_dataset(
        ds: Any,
        split_name: str,
        *,
        feature_key: str,
        source: str,
    ) -> None:
        if ds is None or mamp_encoder is None:
            return
        if not hasattr(ds, "__len__") or len(ds) <= 0:
            return
        pre_bs = max(1, int(config.get("card_mamp_precompute_batch_size", batch_size)))
        tmp_loader = DataLoader(ds, batch_size=pre_bs, shuffle=False, drop_last=False)
        cached_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in tmp_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError(f"Unexpected batch format while precomputing MAMP features for {split_name}.")
                inp = batch[0].to(device).float()
                in_3d = inp[:, :, :, 4:]
                if source == "history":
                    feat = _compute_mamp_feat(in_3d)
                elif source == "coarse":
                    if card_model is None:
                        raise RuntimeError("Two-stage model is not available while precomputing coarse MAMP features.")
                    if diffusion_only:
                        coarse_3d = card_model._zero_coarse_future(in_3d)
                    else:
                        coarse_3d = card_model.coarse(in_3d)
                    feat = _compute_mamp_feat(coarse_3d)
                else:
                    raise ValueError(f"Unknown MAMP precompute source: {source}")
                if feat is None:
                    raise RuntimeError("MAMP encoder is not available during precompute.")
                cached_chunks.append(feat.cpu())
        if not cached_chunks:
            return
        cached = torch.cat(cached_chunks, dim=0).float().contiguous()
        if int(cached.shape[0]) != len(ds):
            raise RuntimeError(
                f"Precomputed MAMP features mismatch for {split_name}: got {cached.shape[0]}, expected {len(ds)}."
            )
        _set_dataset_aux_feature(ds, feature_key, cached)
        print(f"[Info] Precomputed MAMP {feature_key} features for {split_name}: {tuple(cached.shape)}")

    def _extract_cached_mamp_features(aux_payload: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        hist_feat: Optional[torch.Tensor] = None
        coarse_feat: Optional[torch.Tensor] = None
        if torch.is_tensor(aux_payload):
            hist_feat = aux_payload
        elif isinstance(aux_payload, dict):
            h = aux_payload.get("history")
            c = aux_payload.get("coarse")
            if torch.is_tensor(h):
                hist_feat = h
            if torch.is_tensor(c):
                coarse_feat = c
        return hist_feat, coarse_feat

    load_model_path = str(config.get("load_model_path", "")).strip()
    if load_model_path:
        if os.path.exists(load_model_path):
            try:
                try:
                    payload = torch.load(load_model_path, map_location=device, weights_only=True)
                except Exception:
                    payload = torch.load(load_model_path, map_location=device, weights_only=False)
                if isinstance(payload, dict):
                    if "best_model_state" in payload:
                        state = payload["best_model_state"]
                    elif "model_state" in payload:
                        state = payload["model_state"]
                    else:
                        state = payload
                else:
                    state = payload
                _load_state(state)
                print(f"[Info] Loaded model checkpoint from {load_model_path}")
            except Exception as exc:
                print(f"[Warning] Failed to load checkpoint from {load_model_path}: {exc}")
        else:
            print(f"[Warning] Checkpoint not found at {load_model_path}")

    if model == "card" and card_use_mamp_condition and mamp_encoder is not None:
        _precompute_mamp_features_for_dataset(
            getattr(loader, "dataset", None), "train", feature_key="history", source="history"
        )
        _precompute_mamp_features_for_dataset(
            getattr(val_loader, "dataset", None), "val", feature_key="history", source="history"
        )
        _precompute_mamp_features_for_dataset(
            getattr(test_loader, "dataset", None), "test", feature_key="history", source="history"
        )

    def _evaluate_loader(
        eval_loader: Optional[DataLoader],
        *,
        collect_examples: bool = False,
        restore_train: bool = True,
        save_examples: bool = False,
        ) -> Optional[Dict[str, float]]:
        if eval_loader is None:
            return None
        total_loss = 0.0
        total_loss_norm = 0.0
        total_score = 0.0
        total_velocity_mae = 0.0
        total_samples = 0
        collect_for_saving = bool(collect_examples and save_examples)
        compute_humanmac_metrics = bool(config.get("compute_humanmac_metrics", False))
        use_oracle_mpjpe = bool(config.get("card_eval_oracle_mpjpe", False))
        eval_examples_path_cfg = config.get("eval_examples_path")
        if eval_examples_path_cfg:
            eval_examples_path = str(eval_examples_path_cfg)
            eval_examples_dir = os.path.dirname(eval_examples_path) or "."
        else:
            eval_examples_dir = str(
                config.get(
                    "eval_examples_dir",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"),
                )
            )
            eval_examples_path = os.path.join(eval_examples_dir, f"{model}_eval_samples.npz")
        collected_preds: List[torch.Tensor] = []
        collected_tgts: List[torch.Tensor] = []
        collected_obs: List[torch.Tensor] = []
        collected_allk_preds: List[torch.Tensor] = []
        humanmac_pred_batches: List[torch.Tensor] = []
        humanmac_tgt_batches: List[torch.Tensor] = []
        humanmac_context_batches: List[torch.Tensor] = []
        collect_card_tries = (
            bool(config.get("card_eval_collect_all", False))
            and model == "card"
            and card_phase != "coarse"
            and card_eval_best_of_k > 1
        )
        if save_eval_examples and not (
            model == "card" and card_phase != "coarse" and card_eval_best_of_k > 1
        ):
            print("[Info] Saving all candidate predictions is only supported for card with best_of_k > 1.")
        per_try_loss: Optional[List[float]] = None
        per_try_loss_norm: Optional[List[float]] = None
        if collect_card_tries:
            per_try_loss = [0.0 for _ in range(card_eval_best_of_k)]
            per_try_loss_norm = [0.0 for _ in range(card_eval_best_of_k)]

        _set_mode(False)
        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(eval_loader):
                    if isinstance(batch, (list, tuple)):
                        if len(batch) < 2:
                            raise ValueError("Loader batch must provide input and target tensors.")
                        inp, out = batch[0], batch[1]
                        norm_factor = batch[2] if len(batch) > 2 else None
                        cached_aux = batch[3] if len(batch) > 3 else None
                        cached_mamp_feat, cached_mamp_feat_coarse = _extract_cached_mamp_features(cached_aux)
                    else:
                        raise TypeError("Expected batch from loader to be a tuple/list.")

                    inp = inp.to(device=device, dtype=torch.float32, non_blocking=True)
                    out = out.to(device=device, dtype=torch.float32, non_blocking=True)
                    if norm_factor is not None:
                        norm_factor_t = norm_factor.to(
                            device=device,
                            dtype=torch.float32,
                            non_blocking=True,
                        ).view(-1)
                    else:
                        norm_factor_t = torch.ones(inp.size(0), device=device)

                    tgt_3d = out[:, :, :, 4:]
                    tgt_vel = out[:, :, :, 3]
                    in_3d = inp[:, :, :, 4:]
                    humanmac_candidates_batch: Optional[torch.Tensor] = None
                    card_mamp_feat = None
                    coarse_future_for_condition = None
                    if model == "card" and card_phase != "coarse":
                        if diffusion_only and card_use_mamp_condition_coarse:
                            coarse_future_for_condition = card_model._zero_coarse_future(in_3d)
                        elif card_use_mamp_condition_coarse:
                            # Cache coarse prediction once per eval batch and reuse for all sampled candidates.
                            coarse_future_for_condition = card_model.coarse(in_3d)
                        card_mamp_hist_feat = None
                        card_mamp_coarse_feat = None
                        if card_use_mamp_condition:
                            if cached_mamp_feat is not None:
                                card_mamp_hist_feat = cached_mamp_feat.to(
                                    device=device,
                                    dtype=torch.float32,
                                    non_blocking=True,
                                )
                            else:
                                card_mamp_hist_feat = _compute_mamp_feat(in_3d)
                        if card_use_mamp_condition_coarse:
                            if cached_mamp_feat_coarse is not None:
                                card_mamp_coarse_feat = cached_mamp_feat_coarse.to(
                                    device=device,
                                    dtype=torch.float32,
                                    non_blocking=True,
                                )
                            else:
                                card_mamp_coarse_feat = _compute_mamp_feat(coarse_future_for_condition)
                        card_mamp_feat = _merge_mamp_features(card_mamp_hist_feat, card_mamp_coarse_feat)

                    if model == "simlpe_dct":
                        recons = simlpe_dct(in_3d)
                    elif model == "card":
                        if card_phase == "coarse":
                            recons = card_model.coarse(in_3d)
                        else:
                            selection_k = max(1, int(num_candidates))
                            humanmac_k = selection_k if compute_humanmac_metrics else 1
                            if selection_k <= 1 and humanmac_k <= 1:
                                recons = card_model.predict(
                                    in_3d,
                                    mamp_feat=card_mamp_feat,
                                    coarse_future=coarse_future_for_condition,
                                    deterministic=True,
                                )
                                if compute_humanmac_metrics:
                                    humanmac_candidates_batch = recons.unsqueeze(0)
                            else:
                                all_recons: List[torch.Tensor] = []
                                total_candidate_samples = max(selection_k, humanmac_k)
                                for sample_idx in range(total_candidate_samples):
                                    sample_seed = int(batch_idx * 1000003 + sample_idx)
                                    sampled_recons, sampled_score = card_model.predict(
                                        in_3d,
                                        mamp_feat=card_mamp_feat,
                                        coarse_future=coarse_future_for_condition,
                                        deterministic=False,
                                        seed=sample_seed,
                                        return_score=True,
                                    )
                                    all_recons.append(sampled_recons)
                                    if (
                                        sample_idx < selection_k
                                        and collect_card_tries
                                        and per_try_loss is not None
                                        and per_try_loss_norm is not None
                                    ):
                                        mpjpe_terms_try = torch.norm(sampled_recons - tgt_3d, dim=-1)
                                        per_sample_mpjpe_try = mpjpe_terms_try.mean(dim=(1, 2))
                                        per_sample_mpjpe_norm_try = per_sample_mpjpe_try * norm_factor_t
                                        per_try_loss[sample_idx] += float(per_sample_mpjpe_try.sum().item())
                                        per_try_loss_norm[sample_idx] += float(per_sample_mpjpe_norm_try.sum().item())
                                if not all_recons:
                                    raise RuntimeError("Failed to generate diffusion samples during evaluation.")
                                if compute_humanmac_metrics:
                                    humanmac_candidates_batch = torch.stack(all_recons[:humanmac_k], dim=0)
                                if selection_k <= 1:
                                    recons = card_model.predict(
                                        in_3d,
                                        mamp_feat=card_mamp_feat,
                                        coarse_future=coarse_future_for_condition,
                                        deterministic=True,
                                    )
                                else:
                                    stacked_recons = torch.stack(all_recons[:selection_k], dim=0)  # (K, B, T_out, N, 3)
                                    if use_oracle_mpjpe:
                                        oracle_terms = torch.norm(stacked_recons - tgt_3d.unsqueeze(0), dim=-1)
                                        oracle_mpjpe = oracle_terms.mean(dim=(2, 3))  # (K, B)
                                        best_idx = oracle_mpjpe.argmin(dim=0)
                                        batch_selector = torch.arange(stacked_recons.size(1), device=stacked_recons.device)
                                        recons = stacked_recons[best_idx, batch_selector]
                                    else:
                                        contrastive_scorer = _load_contrastive_model()
                                        if contrastive_scorer is None:
                                            raise RuntimeError("Contrastive scorer requested but not available.")
                                        from models.contrastive.contrastive_score import rank_diffusion_samples
                                        y_candidates = stacked_recons.permute(1, 0, 2, 3, 4)  # (B, K, T_out, N, 3)
                                        recons, _ = rank_diffusion_samples(
                                            contrastive_scorer,
                                            in_3d,
                                            y_candidates,
                                            device=str(device),
                                        )
                                if collect_for_saving and selection_k > 1:
                                    collected_allk_preds.append(stacked_recons[:, 0].detach().cpu())
                    if compute_humanmac_metrics:
                        if humanmac_candidates_batch is None:
                            humanmac_candidates_batch = recons.unsqueeze(0)
                        humanmac_pred_batches.append(humanmac_candidates_batch.detach().cpu())
                        humanmac_tgt_batches.append(tgt_3d.detach().cpu())
                        humanmac_context_batches.append(in_3d.detach().cpu())

                    mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)
                    per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
                    per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t

                    total_loss += float(per_sample_mpjpe.sum(dtype=torch.float32).item())
                    total_loss_norm += float(per_sample_mpjpe_norm.sum().item())
                    last_pose_eval = in_3d[:, -1, :, :]
                    pred_vel_mag_eval = _velocity_magnitude(recons, last_pose_eval)
                    vel_mae_eval = (pred_vel_mag_eval - tgt_vel).abs().mean(dim=(1, 2))
                    total_velocity_mae += float(vel_mae_eval.sum(dtype=torch.float32).item())
                    score_val = U.weighted_joint_loss(joint_weights, recons, tgt_3d, metric='mae')
                    total_score += float(score_val.item())
                    total_samples += inp.size(0)

                    if collect_for_saving:
                        collected_obs.append(in_3d[0].detach().cpu())
                        collected_preds.append(recons[0].detach().cpu())
                        collected_tgts.append(tgt_3d[0].detach().cpu())
        finally:
            if restore_train:
                _set_mode(True)

        if total_samples == 0:
            return {
                "mpjpe": float("nan"),
                "mpjpe_norm": float("nan"),
                "weighted_mae": float("nan"),
                "samples": 0,
            }

        avg_loss = total_loss / total_samples
        avg_loss_norm = total_loss_norm / total_samples
        avg_score = total_score / max(1, total_samples)
        avg_velocity_mae = total_velocity_mae / max(1, total_samples)

        if collect_for_saving and collected_preds and collected_tgts and collected_obs:
            os.makedirs(eval_examples_dir, exist_ok=True)
            merged_pred_all_k = None
            if collected_allk_preds:
                merged_pred_all_k = torch.stack(collected_allk_preds, dim=0)
            save_eval_samples_npz(
                eval_examples_path,
                obs=torch.stack(collected_obs, dim=0),
                target=torch.stack(collected_tgts, dim=0),
                pred=torch.stack(collected_preds, dim=0),
                pred_all=merged_pred_all_k,
                metadata={
                    "model": model,
                    "dataset": config.get("dataset"),
                    "action_filter": config.get("action_filter", ""),
                    "num_candidates": int(card_eval_best_of_k),
                },
            )

        metrics = {
            "mpjpe": float(avg_loss),
            "mpjpe_norm": float(avg_loss_norm),
            "weighted_mae": float(avg_score),
            "velocity_mae": float(avg_velocity_mae),
            "samples": float(total_samples),
        }
        if compute_humanmac_metrics and humanmac_pred_batches and humanmac_tgt_batches and humanmac_context_batches:
            humanmac_metrics = _compute_humanmac_metrics(
                torch.cat(humanmac_pred_batches, dim=1),
                torch.cat(humanmac_tgt_batches, dim=0),
                torch.cat(humanmac_context_batches, dim=0),
                threshold=float(config.get("humanmac_multimodal_threshold", 0.5)),
            )
            metrics.update(humanmac_metrics)
        if collect_card_tries and per_try_loss is not None and per_try_loss_norm is not None:
            per_try_mpjpe = [val / max(1, total_samples) for val in per_try_loss]
            per_try_mpjpe_norm = [val / max(1, total_samples) for val in per_try_loss_norm]
            metrics["mpjpe_by_try"] = per_try_mpjpe
            metrics["mpjpe_norm_by_try"] = per_try_mpjpe_norm
        return metrics

    # Optional LR scheduler that reduces LR on plateau of the monitored loss
    use_lr_scheduler = bool(config.get("use_lr_scheduler", True))
    scheduler = None
    if optimizer is not None and use_lr_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(config.get("lr_factor", 0.5)),
            patience=int(config.get("lr_patience", 5)),
            threshold=float(config.get("lr_threshold", 1e-4)),
            cooldown=int(config.get("lr_cooldown", 0)),
            min_lr=float(config.get("lr_min", 0.0))
        )

    # Compute joint weights for scoring (not used in backprop)
    joint_weights = U.compute_node_motion_weights(loader, device=device, normalize='sum')

    n_train_windows = len(getattr(loader, "dataset", []))
    print(f"Training on {n_train_windows} windows")
    print(f"Config: input_n={input_n}, output_n={output_n}, hidden_size={hidden_size}, batch_size={batch_size}, model={model}")

    best_test_mpjpe = None
    best_test_mpjpe_norm = None
    best_test_epoch = None
    best_test_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    last_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_test_velocity_mae = None

    best_train_mpjpe = None
    best_train_mpjpe_norm = None
    best_train_epoch = None
    best_train_velocity_mae = None

    total_epochs = train_epochs
    if model == "card":
        total_epochs = train_epochs + max(0, card_diffusion_epochs)
    early_stop_enabled = bool(config.get("early_stopping_enabled", False))
    early_stop_patience = max(1, int(config.get("early_stopping_patience", 20)))
    early_stop_min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    early_stop_warmup = max(0, int(config.get("early_stopping_warmup", 0)))
    early_stop_monitor = str(config.get("early_stopping_monitor", "auto")).strip().lower()
    early_stop_reset_on_phase_change = bool(config.get("early_stopping_reset_on_phase_change", True))
    early_stop_best: Optional[float] = None
    early_stop_bad_epochs = 0
    # Epoch index (1-based) where the current early-stop stage starts.
    # Warmup is applied relative to this boundary.
    early_stop_stage_start_epoch = 1

    for epoch in range(1, total_epochs + 1):
        if model == "card" and card_phase == "coarse":
            if card_diffusion_epochs > 0 and epoch == train_epochs + 1:
                if maybe_save_coarse_model is not None:
                    maybe_save_coarse_model()
                params = _set_card_phase("diffusion")
                optimizer = torch.optim.Adam(params, lr=card_diffusion_lr)
                card_coarse_diffusion_group_added = False
                if use_lr_scheduler:
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=float(config.get("lr_factor", 0.5)),
                        patience=int(config.get("lr_patience", 5)),
                        threshold=float(config.get("lr_threshold", 1e-4)),
                        cooldown=int(config.get("lr_cooldown", 0)),
                        min_lr=float(config.get("lr_min", 0.0))
                    )
                if card_use_mamp_condition_coarse and mamp_encoder is not None:
                    _set_mode(False)
                    _precompute_mamp_features_for_dataset(
                        getattr(loader, "dataset", None), "train", feature_key="coarse", source="coarse"
                    )
                    _precompute_mamp_features_for_dataset(
                        getattr(val_loader, "dataset", None), "val", feature_key="coarse", source="coarse"
                    )
                    _precompute_mamp_features_for_dataset(
                        getattr(test_loader, "dataset", None), "test", feature_key="coarse", source="coarse"
                    )
                if early_stop_reset_on_phase_change:
                    early_stop_best = None
                    early_stop_bad_epochs = 0
                    early_stop_stage_start_epoch = epoch
                print(f"[Info] Switched to card diffusion training for {card_diffusion_epochs} epochs.")
        if (
            model == "card"
            and card_phase == "diffusion"
            and card_train_coarse_in_diffusion
            and not card_coarse_diffusion_group_added
        ):
            diffusion_stage_epoch = epoch - train_epochs
            if diffusion_stage_epoch > card_diffusion_coarse_warmup_epochs:
                coarse_params = [p for p in card_model.coarse.parameters()]
                for p in coarse_params:
                    p.requires_grad = True
                coarse_params = [p for p in coarse_params if p.requires_grad]
                if coarse_params:
                    if optimizer is None:
                        raise RuntimeError("Optimizer is not initialized for card diffusion phase.")
                    coarse_lr = card_diffusion_lr * 0.1
                    optimizer.add_param_group({"params": coarse_params, "lr": coarse_lr})
                    card_coarse_diffusion_group_added = True
                    print(
                        f"[Info] Unfroze coarse module during diffusion at epoch {epoch} "
                        f"(diffusion_epoch={diffusion_stage_epoch}, warmup={card_diffusion_coarse_warmup_epochs}, "
                        f"coarse_lr={coarse_lr:.6g}, diffusion_lr={card_diffusion_lr:.6g})."
                    )
        _set_mode(True)
        running = 0.0
        score_running = 0.0
        mpjpe_running = 0.0
        mpjpe_norm_running = 0.0
        vel_running = 0.0
        diffusion_running = 0.0
        n_seen = 0
        t0 = time.time()
        for it, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Loader batch must provide input and target tensors.")
                inp, out = batch[0], batch[1]
                norm_factor = batch[2] if len(batch) > 2 else None
                cached_aux = batch[3] if len(batch) > 3 else None
                cached_mamp_feat, cached_mamp_feat_coarse = _extract_cached_mamp_features(cached_aux)
            else:
                raise TypeError("Expected batch from loader to be a tuple/list.")
            # Shapes: inp (B, T_in, N, 7), out (B, T_out, N, 7)
            inp = inp.to(device=device, dtype=torch.float32, non_blocking=True)
            out = out.to(device=device, dtype=torch.float32, non_blocking=True)
            if norm_factor is not None:
                norm_factor_t = norm_factor.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).view(-1)
            else:
                norm_factor_t = torch.ones(inp.size(0), device=device)

            # Prepare inputs/targets as in prediction_model.py
            tgt_vel = out[:, :, :, 3]
            in_3d = inp[:, :, :, 4:]
            tgt_3d = out[:, :, :, 4:]
            diffusion_loss = None

            if model == "simlpe_dct":
                optimizer.zero_grad(set_to_none=True)
                recons = simlpe_dct(in_3d)

            elif model == "card":
                optimizer.zero_grad(set_to_none=True)
                if card_phase == "diffusion":
                    card_mamp_hist_feat = None
                    card_mamp_coarse_feat = None
                    coarse_grad_active = bool(
                        card_train_coarse_in_diffusion and card_coarse_diffusion_group_added
                    )
                    if diffusion_only:
                        coarse_future_for_condition = (
                            card_model._zero_coarse_future(in_3d)
                            if card_use_mamp_condition_coarse
                            else None
                        )
                    elif card_use_mamp_condition_coarse:
                        # Cache coarse prediction once per train batch during diffusion phase.
                        if coarse_grad_active:
                            coarse_future_for_condition = card_model.coarse(in_3d)
                        else:
                            with torch.no_grad():
                                coarse_future_for_condition = card_model.coarse(in_3d)
                    else:
                        coarse_future_for_condition = None
                    if card_use_mamp_condition:
                        if cached_mamp_feat is not None:
                            card_mamp_hist_feat = cached_mamp_feat.to(
                                device=device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                        else:
                            card_mamp_hist_feat = _compute_mamp_feat(in_3d)
                    if card_use_mamp_condition_coarse:
                        # Coarse-conditioned MAMP branch is always computed without gradients.
                        coarse_for_mamp = coarse_future_for_condition.detach()
                        if cached_mamp_feat_coarse is not None:
                            card_mamp_coarse_feat = cached_mamp_feat_coarse.to(
                                device=device,
                                dtype=torch.float32,
                                non_blocking=True,
                            )
                        else:
                            card_mamp_coarse_feat = _compute_mamp_feat(coarse_for_mamp)
                    card_mamp_feat = _merge_mamp_features(card_mamp_hist_feat, card_mamp_coarse_feat)
                    diffusion_loss, coarse_pred = card_model.diffusion_loss(
                        in_3d,
                        tgt_3d,
                        mamp_feat=card_mamp_feat,
                        coarse_future=coarse_future_for_condition,
                        allow_coarse_grad=coarse_grad_active,
                    )
                    recons = coarse_pred
                else:
                    recons = card_model.coarse(in_3d)
            supervised_tgt_3d = tgt_3d
            if (
                model == "card"
                and card_phase == "coarse"
                and coarse_target_lowpass_only
            ):
                with torch.no_grad():
                    supervised_tgt_3d = card_model.coarse.lowpass_future_target(in_3d, tgt_3d)
            if model == "card" and card_phase == "diffusion":
                if diffusion_loss is None:
                    raise RuntimeError("Expected diffusion_loss to be set for card.")
                loss = diffusion_loss
            else:
                mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)  # (B, T_out, N)
                per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
                per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t
                mpjpe_loss = per_sample_mpjpe.mean()
                mpjpe_running += float(per_sample_mpjpe.sum(dtype=torch.float32).item())
                mpjpe_norm_running += float(per_sample_mpjpe_norm.sum().item())
                # Optional bone-length loss on hand skeleton
                if bone_loss_weight > 0:
                    supervised_mpjpe_loss = torch.norm(recons - supervised_tgt_3d, dim=-1).mean()
                    bl_loss = U.bone_length_loss_edges(recons, supervised_tgt_3d, edges_t)
                    loss = supervised_mpjpe_loss + float(bone_loss_weight) * bl_loss
                else:
                    loss = torch.norm(recons - supervised_tgt_3d, dim=-1).mean()
                last_observed_pose = in_3d[:, -1, :, :]
                pred_velocity_mag = _velocity_magnitude(recons, last_observed_pose)
                tgt_velocity_mag = _velocity_magnitude(supervised_tgt_3d, last_observed_pose)
                vel_mae_per_sample = (pred_velocity_mag - tgt_velocity_mag).abs().mean(dim=(1, 2))
                vel_loss = vel_mae_per_sample.mean()
                if velocity_loss_weight > 0:
                    loss = loss + velocity_loss_weight * vel_loss
                vel_running += float(vel_mae_per_sample.sum(dtype=torch.float32).item())
            if model == "card" and card_phase == "diffusion":
                with torch.no_grad():
                    mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)
                    per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
                    per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t
                    mpjpe_running += float(per_sample_mpjpe.sum(dtype=torch.float32).item())
                    mpjpe_norm_running += float(per_sample_mpjpe_norm.sum().item())
                    last_observed_pose = in_3d[:, -1, :, :]
                    pred_velocity_mag = _velocity_magnitude(recons, last_observed_pose)
                    tgt_velocity_mag = _velocity_magnitude(supervised_tgt_3d, last_observed_pose)
                    vel_mae_per_sample = (pred_velocity_mag - tgt_velocity_mag).abs().mean(dim=(1, 2))
                    vel_running += float(vel_mae_per_sample.sum(dtype=torch.float32).item())
            if diffusion_loss is not None:
                diffusion_running += float(diffusion_loss.item()) * inp.size(0)

            # Abort immediately if loss becomes NaN/Inf during any batch
            if not torch.isfinite(loss):
                print("[NaN detected] Loss is NaN/Inf; aborting training and returning NaN metrics.")
                return {
                    "test_mpjpe": float('nan'),
                    "test_weighted_mae": float('nan'),
                }

            if model in ("simlpe_dct", "card"):
                loss.backward()
                max_norm = float(config.get("gradient_clip", 5.0))
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
                optimizer.step()

            running += float(loss.item()) * inp.size(0)
            # Scoring loss (weighted joint MAE) as in test_prediction.py (not used for backprop)
            with torch.no_grad():
                score_val = U.weighted_joint_loss(joint_weights, recons, tgt_3d, metric='mae')
                score_running += float(score_val.item())
            n_seen += inp.size(0)

        epoch_loss = running / max(1, n_seen)
        dt = time.time() - t0
        score_avg = score_running / max(1, n_seen)
        mpjpe_avg = mpjpe_running / max(1, n_seen)
        mpjpe_norm_avg = mpjpe_norm_running / max(1, n_seen)
        vel_avg = vel_running / max(1, n_seen)
        diffusion_avg = diffusion_running / max(1, n_seen)
        objective_train_loss = diffusion_avg if (model == "card" and card_phase == "diffusion") else epoch_loss
        vel_clause = f"| vel_mae={vel_avg:.6f} "
        phase_clause = ""
        if model == "card":
            phase_clause = f"| phase={card_phase} | diff_loss={diffusion_avg:.6f}"
        print(
            f"[Epoch {epoch:03d}] loss={objective_train_loss:.6f} | mpjpe={mpjpe_avg:.6f} | "
            f"mpjpe_norm={mpjpe_norm_avg:.6f}{vel_clause}"
            f"{phase_clause} | score(mae-wj)={score_avg:.6f} | time={dt:.1f}s | lr={optimizer.param_groups[0]['lr'] if optimizer is not None else 'n/a'}"
        )

        if best_train_mpjpe is None or mpjpe_avg < best_train_mpjpe:
            best_train_mpjpe = mpjpe_avg
            best_train_mpjpe_norm = mpjpe_norm_avg
            best_train_epoch = epoch
        if best_train_velocity_mae is None or vel_avg < best_train_velocity_mae:
            best_train_velocity_mae = vel_avg

    # -----------------
    # Test/Evaluation
    # -----------------
        avg_loss = None
        avg_loss_norm = None
        avg_score = None
        avg_velocity_mae = None
        total_samples = 0
        test_metrics = None
        if model != "card" and test_loader is not None:
            test_metrics = _evaluate_loader(test_loader, collect_examples=False, save_examples=save_eval_examples)
        if test_metrics is not None:
            avg_loss = float(test_metrics["mpjpe"])
            avg_loss_norm = float(test_metrics["mpjpe_norm"])
            avg_score = float(test_metrics["weighted_mae"])
            avg_velocity_mae = float(test_metrics.get("velocity_mae", float("nan")))
            total_samples = int(test_metrics["samples"])
            vel_print = "nan"
            if math.isfinite(avg_velocity_mae):
                vel_print = f"{avg_velocity_mae:.6f}"
            print(
                f"[Test] loss={avg_loss:.6f} | score(mae-wj)={avg_score:.6f} | loss(norm)={avg_loss_norm:.6f} | "
                f"vel_mae={vel_print} | samples={total_samples}"
            )

            if best_test_mpjpe is None or avg_loss < best_test_mpjpe:
                best_test_mpjpe = avg_loss
                best_test_mpjpe_norm = avg_loss_norm
                best_test_epoch = epoch
                best_test_state = _capture_state()
                print(f"[Info] New best test MPJPE {best_test_mpjpe:.6f} at epoch {epoch}")

        if log_wandb and wandb_handle is not None:
            wandb_metrics: Dict[str, float] = {}
            _maybe_add_metric(wandb_metrics, "epoch", float(epoch))
            _maybe_add_metric(wandb_metrics, "train/loss", objective_train_loss)
            _maybe_add_metric(wandb_metrics, "train/mpjpe", mpjpe_avg)
            _maybe_add_metric(wandb_metrics, "train/mpjpe_norm", mpjpe_norm_avg)
            _maybe_add_metric(wandb_metrics, "train/weighted_mae", score_avg)
            _maybe_add_metric(wandb_metrics, "train/velocity_mae", vel_avg)
            if model == "card":
                _maybe_add_metric(wandb_metrics, "train/card_phase", 0.0 if card_phase == "coarse" else 1.0)
                _maybe_add_metric(wandb_metrics, "train/diffusion_loss", diffusion_avg)
            _maybe_add_metric(wandb_metrics, "train/time_s", dt)
            lr_value = optimizer.param_groups[0]['lr'] if optimizer is not None else None
            _maybe_add_metric(wandb_metrics, "train/lr", lr_value)
            if avg_loss is not None:
                _maybe_add_metric(wandb_metrics, "test/loss", avg_loss)
                _maybe_add_metric(wandb_metrics, "test/loss_norm", avg_loss_norm)
                _maybe_add_metric(wandb_metrics, "test/weighted_mae", avg_score)
                if avg_velocity_mae is not None:
                    _maybe_add_metric(wandb_metrics, "test/velocity_mae", avg_velocity_mae)
                _maybe_add_metric(wandb_metrics, "test/samples", float(total_samples))
            if wandb_metrics:
                wandb_handle.log(wandb_metrics, step=epoch)

        # Step LR scheduler based on validation loss if enabled
        if avg_loss is not None:
            if scheduler is not None:
                # Prefer validation loss when available
                scheduler.step(float(avg_loss))
        else:
            # Step LR scheduler based on training loss if no validation loader
            if scheduler is not None:
                scheduler.step(float(objective_train_loss))

        if early_stop_enabled:
            if early_stop_monitor == "test_mpjpe":
                monitored = avg_loss
            elif early_stop_monitor == "train_mpjpe":
                monitored = mpjpe_avg
            elif early_stop_monitor in {"train_loss", "loss"}:
                monitored = objective_train_loss
            else:
                monitored = avg_loss if avg_loss is not None else objective_train_loss
            if monitored is None or not math.isfinite(float(monitored)):
                print("[EarlyStop] Monitored value is not finite; skipping early-stop check for this epoch.")
            else:
                early_stop_stage_epoch = epoch - early_stop_stage_start_epoch + 1
                if early_stop_stage_epoch > early_stop_warmup:
                    monitored_f = float(monitored)
                    improved = early_stop_best is None or (monitored_f < (float(early_stop_best) - early_stop_min_delta))
                    if improved:
                        early_stop_best = monitored_f
                        early_stop_bad_epochs = 0
                    else:
                        early_stop_bad_epochs += 1
                        if early_stop_bad_epochs >= early_stop_patience:
                            if model == "card" and card_phase == "coarse" and card_diffusion_epochs > 0:
                                train_epochs = epoch
                                early_stop_best = None
                                early_stop_bad_epochs = 0
                                print(
                                    f"[EarlyStop] Coarse stage stopped at epoch {epoch}; "
                                    "switching to card diffusion stage."
                                )
                            else:
                                print(
                                    f"[EarlyStop] Stopping at epoch {epoch} "
                                    f"(monitor={early_stop_monitor or 'auto'}, best={early_stop_best:.6f}, "
                                    f"current={monitored_f:.6f}, patience={early_stop_patience}, min_delta={early_stop_min_delta})."
                                )
                                break

    last_state = _capture_state()

    if maybe_save_coarse_model is not None:
        maybe_save_coarse_model()

    val_metrics_best = None
    if val_loader is not None:
        state_snapshot = _capture_state()
        if best_test_state is not None:
            _load_state(best_test_state)
        val_metrics_best = _evaluate_loader(val_loader, collect_examples=True, restore_train=False, save_examples=save_eval_examples)
        if state_snapshot is not None:
            _load_state(state_snapshot)

    if val_metrics_best is not None:
        val_mpjpe_best = float(val_metrics_best["mpjpe"])
        val_mpjpe_norm_best = float(val_metrics_best["mpjpe_norm"])
        val_weighted_mae_best = float(val_metrics_best["weighted_mae"])
        val_velocity_mae_best = float(val_metrics_best.get("velocity_mae", float("nan")))
        val_humanmac_apd_best = float(val_metrics_best.get("humanmac_apd", float("nan")))
        val_humanmac_ade_best = float(val_metrics_best.get("humanmac_ade", float("nan")))
        val_humanmac_fde_best = float(val_metrics_best.get("humanmac_fde", float("nan")))
        val_humanmac_mmade_best = float(val_metrics_best.get("humanmac_mmade", float("nan")))
        val_humanmac_mmfde_best = float(val_metrics_best.get("humanmac_mmfde", float("nan")))
        val_humanmac_cmd_best = float(val_metrics_best.get("humanmac_cmd", float("nan")))
        val_humanmac_fid_best = float(val_metrics_best.get("humanmac_fid", float("nan")))
        val_samples = int(val_metrics_best["samples"])
        ref_epoch = best_test_epoch if best_test_epoch is not None else "final"
        print(
            f"[Validation @ epoch {ref_epoch}] mpjpe={val_mpjpe_best:.6f} | "
            f"mpjpe_norm={val_mpjpe_norm_best:.6f} | "
            f"score(mae-wj)={val_weighted_mae_best:.6f} | vel_mae={val_velocity_mae_best:.6f} | samples={val_samples}"
        )
        if bool(config.get("compute_humanmac_metrics", False)):
            print(
                f"[Validation HumanMAC @ epoch {ref_epoch}] APD={val_humanmac_apd_best:.6f} | "
                f"ADE={val_humanmac_ade_best:.6f} | FDE={val_humanmac_fde_best:.6f} | "
                f"MMADE={val_humanmac_mmade_best:.6f} | MMFDE={val_humanmac_mmfde_best:.6f} | "
                f"CMD={val_humanmac_cmd_best:.6f} | FID={val_humanmac_fid_best:.6f}"
            )
    else:
        val_mpjpe_best = None
        val_mpjpe_norm_best = None
        val_weighted_mae_best = None
        val_velocity_mae_best = None
        val_humanmac_apd_best = None
        val_humanmac_ade_best = None
        val_humanmac_fde_best = None
        val_humanmac_mmade_best = None
        val_humanmac_mmfde_best = None
        val_humanmac_cmd_best = None
        val_humanmac_fid_best = None
        val_samples = 0

    best_model_path = config.get("best_model_path")
    if best_model_path:
        if best_test_state is not None:
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(
                {
                    "best_model_state": best_test_state,
                    "metadata": {
                        "tag": config.get("best_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                    },
                },
                best_model_path,
            )
            print(f"Saved best model checkpoint to {best_model_path}")
        elif model == "card" and last_state is not None:
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(
                {
                    "best_model_state": last_state,
                    "metadata": {
                        "tag": config.get("best_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                        "note": "last_state_after_diffusion",
                    },
                },
                best_model_path,
            )
            print(f"Saved last card checkpoint to {best_model_path}")
        else:
            print(f"Best model checkpoint was requested but no best state was captured for {best_model_path}")

    final_model_path = config.get("final_model_path")
    if final_model_path:
        if last_state is not None:
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            torch.save(
                {
                    "final_model_state": last_state,
                    "metadata": {
                        "tag": config.get("best_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                    },
                },
                final_model_path,
            )
            print(f"Saved final model checkpoint to {final_model_path}")
        else:
            print(f"Final model checkpoint was requested but no final state was captured for {final_model_path}")

    return {
        "train_mpjpe_best": float(best_train_mpjpe) if best_train_mpjpe is not None else None,
        "train_mpjpe_norm_best": float(best_train_mpjpe_norm) if best_train_mpjpe_norm is not None else None,
        "train_best_epoch": int(best_train_epoch) if best_train_epoch is not None else None,
        "train_mpjpe_best": float(best_train_mpjpe) if best_train_mpjpe is not None else None,
        "train_mpjpe_norm_best": float(best_train_mpjpe_norm) if best_train_mpjpe_norm is not None else None,
        "validation_mpjpe_best": val_mpjpe_best if val_loader is not None else None,
        "validation_mpjpe_norm_best": val_mpjpe_norm_best if val_loader is not None else None,
        "validation_humanmac_apd_best": val_humanmac_apd_best if val_loader is not None else None,
        "validation_humanmac_ade_best": val_humanmac_ade_best if val_loader is not None else None,
        "validation_humanmac_fde_best": val_humanmac_fde_best if val_loader is not None else None,
        "validation_humanmac_mmade_best": val_humanmac_mmade_best if val_loader is not None else None,
        "validation_humanmac_mmfde_best": val_humanmac_mmfde_best if val_loader is not None else None,
        "validation_humanmac_cmd_best": val_humanmac_cmd_best if val_loader is not None else None,
        "validation_humanmac_fid_best": val_humanmac_fid_best if val_loader is not None else None,
        "validation_samples": float(val_samples) if val_loader is not None else None,
        "validation_mpjpe_by_try": (
            val_metrics_best.get("mpjpe_by_try") if val_metrics_best is not None else None
        ),
        "validation_mpjpe_norm_by_try": (
            val_metrics_best.get("mpjpe_norm_by_try") if val_metrics_best is not None else None
        ),
        "test_mpjpe_best": best_test_mpjpe if test_loader is not None else None,
        "test_mpjpe_norm_best": best_test_mpjpe_norm if test_loader is not None else None,
        "params": params_count,
    }
