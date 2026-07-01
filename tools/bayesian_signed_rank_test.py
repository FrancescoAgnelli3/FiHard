#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import math
import pickle
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTORANK_ROOT = REPO_ROOT.parent / "autorank"
VENDOR_SPLINE = REPO_ROOT / "vendor" / "splineeqnet"
VENDOR_COMUSION = REPO_ROOT / "vendor" / "comusion"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = ("APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID")
BAYESIAN_METRICS = tuple(metric for metric in METRICS if metric != "APD")


@contextlib.contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(message: str) -> None:
    print(f"[bayes-rank] {message}", flush=True)


def purge_vendor_modules(*module_names: str) -> None:
    for name in module_names:
        for key in list(sys.modules.keys()):
            if key == name or key.startswith(f"{name}."):
                sys.modules.pop(key, None)
    importlib.invalidate_caches()


@contextlib.contextmanager
def force_torch_load_map_location(device: torch.device):
    original_torch_load = torch.load

    def wrapped_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = wrapped_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_comusion_checkpoint(checkpoint_path: Path, device: torch.device):
    with open(checkpoint_path, "rb") as handle:
        with force_torch_load_map_location(device):
            return pickle.load(handle)


def extract_pairwise_posterior_triplet(result, left_label: str, right_label: str) -> Tuple[float, float, float]:
    if result.posterior_matrix is None:
        raise RuntimeError("Bayesian result does not contain a posterior_matrix.")

    if left_label not in result.posterior_matrix.index:
        raise KeyError(f"{left_label!r} not found in posterior_matrix index")
    if right_label not in result.posterior_matrix.columns:
        raise KeyError(f"{right_label!r} not found in posterior_matrix columns")

    direct = result.posterior_matrix.loc[left_label, right_label]
    if isinstance(direct, tuple):
        return tuple(float(x) for x in direct)

    reverse = result.posterior_matrix.loc[right_label, left_label]
    if isinstance(reverse, tuple):
        p_left, p_equal, p_right = (float(x) for x in reverse)
        return p_right, p_equal, p_left

    raise RuntimeError(
        f"Could not extract posterior triplet for pair ({left_label}, {right_label}) from posterior_matrix."
    )


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired test-set metrics and Bayesian signed-rank plots for two checkpoints."
    )
    parser.add_argument(
        "--FiHard-checkpoint",
        type=Path,
        default=Path(
            REPO_ROOT
            / "out"
            / "diffusion_hands_runs"
            / "FiHard"
            / "assembly_pick_up_screwd_20260504_163404"
            / "checkpoints"
            / "final.pt"
        ),
    )
    parser.add_argument(
        "--comusion-checkpoint",
        type=Path,
        default=Path(
            REPO_ROOT
            / "out"
            / "diffusion_hands_runs"
            / "comusion"
            / "assembly_pick_up_screwd_20260504_163404"
            / "checkpoints"
            / "final.pt"
        ),
    )
    parser.add_argument("--dataset", default="assembly")
    parser.add_argument("--action-filter", default="pick_up_screwd")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--humanmac-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rope", type=float, default=0.1)
    parser.add_argument("--rope-mode", default="effsize")
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out" / "bayesian_signed_rank" / "assembly_pick_up_screwd_FiHard_vs_comusion",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> dict:
    experiment_cfg = load_yaml(REPO_ROOT / "configs" / "experiment.yaml")
    preprocessing = dict(experiment_cfg.get("preprocessing", {}))
    data_roots = {
        "assembly": "/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our/",
        "h2o": "/mnt/pve/Turing-Storage2/h2o/",
        "bighands": "/mnt/pve/Turing-Storage2/BigHands/BigHand2.2M/data/",
        "fpha": "/mnt/pve/Turing-Storage2/FPHA/data/",
    }
    user_roots = experiment_cfg.get("data_roots")
    if isinstance(user_roots, dict):
        for key, value in user_roots.items():
            data_roots[str(key).strip().lower()] = str(value)
    dataset_key = str(args.dataset).strip().lower()
    resolved_data_dir = str(args.data_dir) if args.data_dir is not None else data_roots.get(dataset_key, "")
    return {
        "seed": int(experiment_cfg.get("seed", 0) if args.seed is None else args.seed),
        "num_candidates": int(
            experiment_cfg.get("num_candidates", 10) if args.num_candidates is None else args.num_candidates
        ),
        "humanmac_threshold": float(
            experiment_cfg.get("humanmac_multimodal_threshold", 7.0)
            if args.humanmac_threshold is None
            else args.humanmac_threshold
        ),
        "preprocessing": preprocessing,
        "data_dir": resolved_data_dir,
    }


def compute_multimodal_groups(context_flat: torch.Tensor, threshold: float) -> List[torch.Tensor]:
    if context_flat.ndim > 2:
        context_flat = context_flat.reshape(context_flat.shape[0], -1)
    pairwise = torch.cdist(context_flat.cpu(), context_flat.cpu())
    groups: List[torch.Tensor] = []
    for sample_idx in range(pairwise.shape[0]):
        groups.append(torch.nonzero(pairwise[sample_idx] < float(threshold), as_tuple=False).reshape(-1))
    return groups


def compute_per_sample_metric_table(
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    context_flat: torch.Tensor,
    threshold: float,
) -> pd.DataFrame:
    from common.metrics import compute_all_metrics_single, distributional_motion_metrics

    groups = compute_multimodal_groups(context_flat=context_flat, threshold=threshold)
    rows: List[Dict[str, float]] = []
    gt_flat = gt_future.reshape(gt_future.shape[0], gt_future.shape[1], -1).cpu()
    pred_flat = pred_candidates.reshape(pred_candidates.shape[0], pred_candidates.shape[1], pred_candidates.shape[2], -1).cpu()

    for sample_idx in range(gt_flat.shape[0]):
        pred_i = pred_flat[sample_idx]
        gt_i = gt_flat[sample_idx : sample_idx + 1]
        gt_multi = gt_flat[groups[sample_idx]]
        apd, ade, fde, mmade, mmfde = compute_all_metrics_single(pred_i, gt_i, gt_multi)
        dist_metrics = distributional_motion_metrics(pred_candidates[sample_idx], gt_future[sample_idx : sample_idx + 1])
        rows.append(
            {
                "sample_idx": sample_idx,
                "APD": float(apd.item()),
                "ADE": float(ade.item()),
                "FDE": float(fde.item()),
                "MMADE": float(mmade.item()),
                "MMFDE": float(mmfde.item()),
                "CMD": float(dist_metrics["CMD"]),
                "FID": float(dist_metrics["FID"]),
            }
        )
    return pd.DataFrame(rows)


def save_prediction_cache(
    cache_path: Path,
    *,
    pred_candidates: torch.Tensor,
    gt_future: torch.Tensor,
    context: torch.Tensor,
    metadata: Dict[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_candidates": pred_candidates.cpu(),
            "gt_future": gt_future.cpu(),
            "context": context.cpu(),
            "metadata": dict(metadata),
        },
        cache_path,
    )


def load_prediction_cache(cache_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = ("pred_candidates", "gt_future", "context")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Prediction cache missing keys {missing}: {cache_path}")
    return payload


def prediction_cache_metadata(
    *,
    model_name: str,
    checkpoint_path: Path,
    dataset: str,
    action_filter: str,
    runtime_cfg: dict,
) -> Dict[str, object]:
    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset": str(dataset),
        "action_filter": str(action_filter),
        "num_candidates": int(runtime_cfg["num_candidates"]),
        "humanmac_threshold": float(runtime_cfg["humanmac_threshold"]),
        "seed": int(runtime_cfg["seed"]),
    }


def cache_matches(payload: Dict[str, object], expected: Dict[str, object]) -> bool:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True


def build_FiHard_test_loader(dataset: str, action_filter: str, runtime_cfg: dict):
    purge_vendor_modules("config", "data", "datasets", "models", "runner", "train_utils")
    with prepend_sys_path(VENDOR_SPLINE):
        from config import DatasetCfg
        from data import build_datasets, get_dataset_metadata, make_loaders

        metadata = get_dataset_metadata(dataset)
        pp = runtime_cfg["preprocessing"]
        data_dir = str(runtime_cfg.get("data_dir", "") or metadata.get("default_dir", ""))
        ds_cfg = DatasetCfg(
            data_dir=data_dir,
            action_filter=action_filter,
            input_n=int(pp["input_n"]),
            output_n=int(pp["output_n"]),
            stride=int(pp["stride"]),
            time_interp=pp.get("time_interp"),
            window_norm=pp.get("window_norm"),
            batch_size=512,
            eval_batch_mult=int(pp.get("eval_batch_mult", 1)),
            seed=int(runtime_cfg["seed"]),
            wrist_indices=tuple(int(idx) for idx in metadata.get("default_wrist_indices", ())),
            dataset=dataset,
            node_count=int(metadata.get("node_count", 21)),
            edge_index=tuple(metadata.get("edge_index", ())),
            adjacency=tuple(metadata.get("adjacency", ())),
        )
        train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
        _train_loader, _val_loader, test_loader = make_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            ds_cfg.batch_size,
            ds_cfg.seed,
            ds_cfg.eval_batch_mult,
        )
        return ds_cfg, metadata, test_loader


def evaluate_FiHard(
    checkpoint_path: Path,
    dataset: str,
    action_filter: str,
    runtime_cfg: dict,
    device: torch.device,
    cache_path: Path,
    force_recompute: bool,
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    expected_meta = prediction_cache_metadata(
        model_name="FiHard",
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        action_filter=action_filter,
        runtime_cfg=runtime_cfg,
    )
    if cache_path.exists() and not force_recompute:
        payload = load_prediction_cache(cache_path)
        if cache_matches(payload, expected_meta):
            log(f"Loading cached FiHard predictions from {cache_path}")
            pred_candidates = payload["pred_candidates"]
            gt_future = payload["gt_future"]
            context = payload["context"]
            table = compute_per_sample_metric_table(
                pred_candidates=pred_candidates,
                gt_future=gt_future,
                context_flat=context.reshape(context.shape[0], context.shape[1], -1),
                threshold=float(runtime_cfg["humanmac_threshold"]),
            )
            log(f"Twostage metrics loaded from cached predictions for {len(table)} samples")
            return table, {"pred_candidates": pred_candidates, "gt_future": gt_future, "context": context}
        log("Twostage cache metadata mismatch, recomputing predictions")

    log(
        "Preparing FiHard evaluation "
        f"(dataset={dataset}, action_filter={action_filter}, checkpoint={checkpoint_path})"
    )
    ds_cfg, _metadata, test_loader = build_FiHard_test_loader(dataset, action_filter, runtime_cfg)
    log(
        "Twostage test loader ready "
        f"(samples={len(test_loader.dataset)}, batches={len(test_loader)}, num_candidates={runtime_cfg['num_candidates']})"
    )

    purge_vendor_modules("config", "data", "datasets", "models", "runner", "train_utils")
    with prepend_sys_path(VENDOR_SPLINE):
        from models.fihard import CardConfig, CardForecaster
        from data import resolve_FiHard_hand_graph_metadata

        model_cfg = load_yaml(REPO_ROOT / "configs" / "models" / "fihard.yaml")["defaults"]
        tw_cfg = CardConfig(
            input_length=int(ds_cfg.input_n),
            pred_length=int(ds_cfg.output_n),
            num_nodes=int(ds_cfg.node_count),
            hidden_dim=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["gru_layers"]),
            k_low=int(model_cfg["FiHard_k_low"]),
            diffusion_steps=int(model_cfg["FiHard_diffusion_steps"]),
            ddim_steps=int(model_cfg["FiHard_ddim_steps"]),
            isotropic_noise=bool(model_cfg["FiHard_isotropic_noise"]),
            beta_matrix_power=float(model_cfg["FiHard_beta_matrix_power"]),
            beta_matrix_min_rate=float(model_cfg["FiHard_beta_matrix_min_rate"]),
            beta_matrix_max_rate=float(model_cfg["FiHard_beta_matrix_max_rate"]),
            node_covariance_type=str(model_cfg["FiHard_node_covariance_type"]),
            mobility_palm_var=float(model_cfg["FiHard_mobility_palm_var"]),
            mobility_depth1_var=float(model_cfg["FiHard_mobility_depth1_var"]),
            mobility_depth2_var=float(model_cfg["FiHard_mobility_depth2_var"]),
            mobility_depth3plus_var=float(model_cfg["FiHard_mobility_depth3plus_var"]),
            dhalf_gamma=float(model_cfg["FiHard_dhalf_gamma"]),
            learnable_dhalf=bool(model_cfg["FiHard_learnable_dhalf"]),
            graph_laplacian_alpha=float(model_cfg["FiHard_graph_laplacian_alpha"]),
            graph_laplacian_beta=float(model_cfg["FiHard_graph_laplacian_beta"]),
            graph_laplacian_normalized=bool(model_cfg.get("FiHard_graph_laplacian_normalized", True)),
            denoiser_dim=int(model_cfg["FiHard_denoiser_dim"]),
            denoiser_depth=int(model_cfg["FiHard_denoiser_depth"]),
            denoiser_heads=int(model_cfg["FiHard_denoiser_heads"]),
            dropout=float(model_cfg["FiHard_dropout"]),
            freeze_coarse=bool(model_cfg["FiHard_freeze_coarse"]),
            cond_use_history=bool(model_cfg["FiHard_cond_use_history"]),
            cond_use_coarse=bool(model_cfg["FiHard_cond_use_coarse"]),
            allow_no_conditioning=bool(model_cfg["FiHard_allow_no_conditioning"]),
            coarse_target_lowpass_only=bool(model_cfg["FiHard_coarse_target_lowpass_only"]),
            diffusion_only=bool(model_cfg["FiHard_diffusion_only"]),
            simlpe_use_norm=bool(model_cfg["simlpe_use_norm"]),
            simlpe_spatial_fc_only=bool(model_cfg["simlpe_use_spatial_fc_only"]),
            simlpe_mix_spatial_temporal=bool(model_cfg["simlpe_mix_spatial_temporal"]),
            simlpe_norm_axis=str(model_cfg["simlpe_norm_axis"]),
            simlpe_add_last_offset=bool(model_cfg["simlpe_add_last_offset"]),
        )
        graph_meta = resolve_FiHard_hand_graph_metadata(
            ds_cfg.dataset,
            tuple(int(idx) for idx in ds_cfg.wrist_indices),
        )
        model = CardForecaster(
            tw_cfg,
            metadata={
                "wrist_index": int(graph_meta["wrist_index"]),
                "edges": tuple((int(i), int(j)) for i, j in tuple(graph_meta["links"])),
                "empirical_feature_covariance": None,
            },
        ).to(device)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = payload.get("final_model_state", payload) if isinstance(payload, dict) else payload
        if isinstance(state, dict) and "FiHard" in state:
            state = state["FiHard"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[FiHard] load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            )
        model.eval()
        log(f"Loaded FiHard checkpoint on device={device}")

        contexts: List[torch.Tensor] = []
        futures: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        num_candidates = int(runtime_cfg["num_candidates"])
        for batch_idx, batch in enumerate(
            tqdm(
                test_loader,
                desc="FiHard test",
                unit="batch",
                leave=True,
            )
        ):
            inp, out = batch[:2]
            inp = inp.to(device).float()
            out = out.to(device).float()
            in_3d = inp[:, :, :, 4:]
            tgt_3d = out[:, :, :, 4:]
            coarse_future = model._zero_coarse_future(in_3d) if tw_cfg.diffusion_only else model.coarse(in_3d)
            sampled: List[torch.Tensor] = []
            for sample_idx in range(num_candidates):
                sampled_pred, _score = model.predict(
                    in_3d,
                    mamp_feat=None,
                    coarse_future=coarse_future,
                    deterministic=False,
                    seed=int(batch_idx * 1000003 + sample_idx),
                    return_score=True,
                )
                sampled.append(sampled_pred.detach().cpu())
            candidates.append(torch.stack(sampled, dim=1))
            contexts.append(in_3d.detach().cpu())
            futures.append(tgt_3d.detach().cpu())
            if batch_idx == 0:
                log(
                    "Twostage first batch shapes "
                    f"(context={tuple(in_3d.shape)}, target={tuple(tgt_3d.shape)}, candidates={tuple(candidates[-1].shape)})"
                )

    pred_candidates = torch.cat(candidates, dim=0)
    gt_future = torch.cat(futures, dim=0)
    context = torch.cat(contexts, dim=0)
    table = compute_per_sample_metric_table(
        pred_candidates=pred_candidates,
        gt_future=gt_future,
        context_flat=context.reshape(context.shape[0], context.shape[1], -1),
        threshold=float(runtime_cfg["humanmac_threshold"]),
    )
    save_prediction_cache(
        cache_path,
        pred_candidates=pred_candidates,
        gt_future=gt_future,
        context=context,
        metadata=expected_meta,
    )
    log(f"Saved FiHard prediction cache to {cache_path}")
    log(f"Twostage metrics computed for {len(table)} samples")
    return table, {"pred_candidates": pred_candidates, "gt_future": gt_future, "context": context}


def build_comusion_cfg(dataset: str, action_filter: str, runtime_cfg: dict) -> SimpleNamespace:
    model_cfg = load_yaml(REPO_ROOT / "configs" / "models" / "comusion.yaml")["defaults"]
    pp = runtime_cfg["preprocessing"]
    ns = SimpleNamespace()
    ns.model_type = str(model_cfg.get("model_type", "CoMusion"))
    ns.t_his = int(pp["input_n"])
    ns.t_pred = int(pp["output_n"])
    ns.eval_sample_num = int(runtime_cfg["num_candidates"])
    ns.dtype = str(model_cfg.get("dtype", "float32"))
    model_specs = dict(model_cfg.get("model_specs", {}))
    ns.node_n = int(model_specs.get("node_n", 63))
    ns.act = str(model_specs.get("act", "nn.Tanh"))
    ns.dct_dim = int(model_specs.get("dct_dim", 100))
    ns.gcn_dim = int(model_specs.get("gcn_dim", 128))
    ns.gcn_drop = float(model_specs.get("gcn_drop", 0.5))
    ns.inner_stage = int(model_specs.get("inner_stage", 2))
    ns.outer_stage = int(model_specs.get("outer_stage", 3))
    ns.trans_dim = int(model_specs.get("trans_dim", 256))
    ns.trans_drop = float(model_specs.get("trans_drop", 0.1))
    ns.trans_ff_dim = int(model_specs.get("trans_ff_dim", 256))
    ns.trans_num_heads = int(model_specs.get("trans_num_heads", 4))
    ns.trans_num_layers = int(model_specs.get("trans_num_layers", 4))
    diff_specs = dict(model_cfg.get("diff_specs", {}))
    ns.diffuse_steps = int(diff_specs.get("diffuse_steps", 10))
    ns.loss_type = str(diff_specs.get("loss_type", "l1"))
    ns.objective = str(diff_specs.get("objective", "pred_x0"))
    ns.beta_schedule = str(diff_specs.get("beta_schedule", "ours"))
    ns.div_k = int(diff_specs.get("div_k", runtime_cfg["num_candidates"]))
    data_specs = dict(model_cfg.get("data_specs", {}))
    ns.dataset = dataset
    ns.actions = "all"
    ns.augmentation = int(data_specs.get("augmentation", 0))
    ns.stride = int(pp["stride"])
    ns.multimodal_threshold = float(runtime_cfg["humanmac_threshold"])
    ns.humanmac_multimodal_threshold = float(runtime_cfg["humanmac_threshold"])
    ns.data_aug = bool(data_specs.get("data_aug", False))
    ns.rota_prob = float(data_specs.get("rota_prob", 0.0))
    ns.data_dir = str(runtime_cfg.get("data_dir", ""))
    ns.action_filter = action_filter
    ns.eval_batch_mult = int(pp.get("eval_batch_mult", 1))
    ns.time_interp = pp.get("time_interp")
    ns.window_norm = pp.get("window_norm")
    ns.splineeqnet_root = str(VENDOR_SPLINE)
    ns.eval_samples_path = ""
    learn_specs = dict(model_cfg.get("learn_specs", {}))
    ns.train_lr = float(learn_specs.get("train_lr", 1e-4))
    ns.weight_decay = float(learn_specs.get("weight_decay", 0.0))
    ns.train_epoch = int(learn_specs.get("train_epoch", 300))
    ns.sched_policy = str(learn_specs.get("sched_policy", "lambda"))
    ns.num_epoch_fix_lr = int(learn_specs.get("num_epoch_fix_lr", 200))
    ns.batch_size = int(learn_specs.get("batch_size", 256))
    ns.early_stopping_enabled = bool(learn_specs.get("early_stopping_enabled", False))
    ns.early_stopping_patience = int(learn_specs.get("early_stopping_patience", 20))
    ns.early_stopping_min_delta = float(learn_specs.get("early_stopping_min_delta", 1e-4))
    ns.early_stopping_warmup = int(learn_specs.get("early_stopping_warmup", 0))
    ns.early_stopping_monitor = str(learn_specs.get("early_stopping_monitor", "train_loss"))
    ns.loss_weight_scale = float(model_cfg.get("st_loss_specs", {}).get("loss_weight_scale", 10))
    ns.history_weight = float(model_cfg.get("loss_weight_specs", {}).get("history_weight", 1))
    ns.future_weight = float(model_cfg.get("loss_weight_specs", {}).get("future_weight", 1))
    ns.final_checkpoint_path = ""
    ns.remove_model_internals = bool(model_cfg.get("remove_model_internals", False))
    return ns


def evaluate_comusion(
    checkpoint_path: Path,
    dataset: str,
    action_filter: str,
    runtime_cfg: dict,
    device: torch.device,
    cache_path: Path,
    force_recompute: bool,
) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor]]:
    expected_meta = prediction_cache_metadata(
        model_name="comusion",
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        action_filter=action_filter,
        runtime_cfg=runtime_cfg,
    )
    if cache_path.exists() and not force_recompute:
        payload = load_prediction_cache(cache_path)
        if cache_matches(payload, expected_meta):
            log(f"Loading cached comusion predictions from {cache_path}")
            pred_candidates = payload["pred_candidates"]
            gt_future = payload["gt_future"]
            context = payload["context"]
            table = compute_per_sample_metric_table(
                pred_candidates=pred_candidates,
                gt_future=gt_future,
                context_flat=context.reshape(context.shape[0], context.shape[1], -1),
                threshold=float(runtime_cfg["humanmac_threshold"]),
            )
            log(f"CoMusion metrics loaded from cached predictions for {len(table)} samples")
            return table, {"pred_candidates": pred_candidates, "gt_future": gt_future, "context": context}
        log("CoMusion cache metadata mismatch, recomputing predictions")

    log(
        "Preparing comusion evaluation "
        f"(dataset={dataset}, action_filter={action_filter}, checkpoint={checkpoint_path})"
    )
    cfg = build_comusion_cfg(dataset, action_filter, runtime_cfg)
    purge_vendor_modules("models", "utils", "train", "data_utils")
    with prepend_sys_path(VENDOR_COMUSION):
        from train import Trainer, generate_loss_weight
        from models.load_models import get_model
        from models.GaussianDiffusion import GaussianDiffusion

        set_seed(runtime_cfg["seed"])
        model = get_model(cfg).to(device=device, dtype=torch.float32)
        payload = load_comusion_checkpoint(checkpoint_path, device)
        state = payload.get("model_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[comusion] load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
            )
        diffuser = GaussianDiffusion(
            model=model,
            cfg=cfg,
            future_motion_size=(cfg.t_pred, cfg.node_n),
            timesteps=cfg.diffuse_steps,
            loss_type=cfg.loss_type,
            objective=cfg.objective,
            beta_schedule=cfg.beta_schedule,
            history_weight=cfg.history_weight,
            future_weight=cfg.future_weight,
            st_loss_weight=generate_loss_weight(cfg),
        ).to(device=device, dtype=torch.float32)
        trainer = Trainer(
            dataset=None,
            diffusion_model=diffuser,
            cfg=cfg,
            train_batch_size=cfg.batch_size,
            train_lr=cfg.train_lr,
            weight_decay=cfg.weight_decay,
            actions="all",
        )
        trainer.model.eval()
        log(
            "CoMusion test loader ready "
            f"(samples={len(trainer.eval_dataloader.dataset)}, batches={len(trainer.eval_dataloader)}, num_candidates={runtime_cfg['num_candidates']})"
        )
        log(f"Loaded comusion checkpoint on device={device}")

        contexts: List[torch.Tensor] = []
        futures: List[torch.Tensor] = []
        candidates: List[torch.Tensor] = []
        num_candidates = int(runtime_cfg["num_candidates"])
        for batch_idx, batch in enumerate(
            tqdm(
                trainer.eval_dataloader,
                desc="comusion test",
                unit="batch",
                leave=True,
            )
        ):
            data, extra = trainer._batch_to_traj_and_extra(batch)
            data_flat = trainer._flatten_motion(data)
            gt = data_flat[:, trainer.input_n :, :].to(device).to(torch.float32)
            batch_candidates: List[torch.Tensor] = []
            for sample_idx in range(num_candidates):
                set_seed(int(batch_idx * 1000003 + sample_idx))
                pred = trainer.get_prediction(
                    data,
                    extra["act"],
                    sample_num=1,
                    uncond=True,
                    use_ema=True,
                    concat_hist=False,
                ).detach().cpu()
                batch_candidates.append(pred)
            candidates.append(torch.stack(batch_candidates, dim=1).reshape(gt.shape[0], num_candidates, cfg.t_pred, -1, 3))
            contexts.append(data_flat[:, : trainer.input_n, :].reshape(gt.shape[0], trainer.input_n, -1, 3).cpu())
            futures.append(gt.reshape(gt.shape[0], cfg.t_pred, -1, 3).cpu())
            if batch_idx == 0:
                log(
                    "CoMusion first batch shapes "
                    f"(context={tuple(contexts[-1].shape)}, target={tuple(futures[-1].shape)}, candidates={tuple(candidates[-1].shape)})"
                )

    pred_candidates = torch.cat(candidates, dim=0)
    gt_future = torch.cat(futures, dim=0)
    context = torch.cat(contexts, dim=0)
    table = compute_per_sample_metric_table(
        pred_candidates=pred_candidates,
        gt_future=gt_future,
        context_flat=context.reshape(context.shape[0], context.shape[1], -1),
        threshold=float(runtime_cfg["humanmac_threshold"]),
    )
    save_prediction_cache(
        cache_path,
        pred_candidates=pred_candidates,
        gt_future=gt_future,
        context=context,
        metadata=expected_meta,
    )
    log(f"Saved comusion prediction cache to {cache_path}")
    log(f"CoMusion metrics computed for {len(table)} samples")
    return table, {"pred_candidates": pred_candidates, "gt_future": gt_future, "context": context}


def save_metric_artifacts(
    output_dir: Path,
    FiHard_table: pd.DataFrame,
    comusion_table: pd.DataFrame,
    alpha: float,
    rope: float,
    rope_mode: str,
    nsamples: int,
) -> None:
    log(f"Saving Bayesian signed-rank artifacts under {output_dir}")
    with prepend_sys_path(AUTORANK_ROOT):
        from autorank import autorank, create_report

        output_dir.mkdir(parents=True, exist_ok=True)
        paired_dir = output_dir / "paired_metric_tables"
        reports_dir = output_dir / "reports"
        paired_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        combined = FiHard_table[["sample_idx"]].copy()
        for metric in METRICS:
            combined[f"FiHard__{metric}"] = FiHard_table[metric]
            combined[f"comusion__{metric}"] = comusion_table[metric]
        combined.to_csv(output_dir / "all_sample_metrics.csv", index=False)

        for metric in METRICS:
            log(f"Running Bayesian signed-rank test for {metric}")
            paired = pd.DataFrame(
                {
                    "sample_idx": FiHard_table["sample_idx"].to_numpy(),
                    "FiHard": FiHard_table[metric].to_numpy(),
                    "comusion": comusion_table[metric].to_numpy(),
                }
            ).dropna().reset_index(drop=True)
            if len(paired) < 2:
                raise RuntimeError(f"Not enough valid paired samples for metric {metric}: {len(paired)}")
            paired.to_csv(paired_dir / f"{metric.lower()}_paired.csv", index=False)

        log("Skipping Bayesian signed-rank test for APD")
        for metric in tqdm(BAYESIAN_METRICS, desc="bayesian metrics", unit="metric", leave=True):
            log(f"Running Bayesian signed-rank test for {metric}")
            paired = pd.read_csv(paired_dir / f"{metric.lower()}_paired.csv")
            result = autorank(
                paired[["FiHard", "comusion"]],
                alpha=alpha,
                verbose=False,
                order="ascending",
                approach="bayesian",
                rope=rope,
                rope_mode=rope_mode,
                nsamples=nsamples,
                random_state=42,
            )

            report_buffer = io.StringIO()
            with contextlib.redirect_stdout(report_buffer):
                create_report(result)
            with open(reports_dir / f"{metric.lower()}_report.txt", "w", encoding="utf-8") as handle:
                handle.write(report_buffer.getvalue())
            log(f"Saved artifacts for {metric}")


def metrics_csv_path(output_dir: Path) -> Path:
    return output_dir / "all_sample_metrics.csv"


def load_metrics_tables_from_csv(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = metrics_csv_path(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    combined = pd.read_csv(csv_path)
    required_columns = ["sample_idx"]
    for metric in METRICS:
        required_columns.append(f"FiHard__{metric}")
        required_columns.append(f"comusion__{metric}")
    missing = [col for col in required_columns if col not in combined.columns]
    if missing:
        raise RuntimeError(f"Metrics CSV is missing required columns: {missing}")

    FiHard_table = pd.DataFrame({"sample_idx": combined["sample_idx"]})
    comusion_table = pd.DataFrame({"sample_idx": combined["sample_idx"]})
    for metric in METRICS:
        FiHard_table[metric] = combined[f"FiHard__{metric}"]
        comusion_table[metric] = combined[f"comusion__{metric}"]
    return FiHard_table, comusion_table


def main() -> None:
    args = parse_args()
    runtime_cfg = resolve_runtime_config(args)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"Starting evaluation (dataset={args.dataset}, action_filter={args.action_filter}, "
        f"data_dir={runtime_cfg.get('data_dir', '')}, device={device}, num_candidates={runtime_cfg['num_candidates']})"
    )
    data_dir = Path(str(runtime_cfg.get("data_dir", "") or "")).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            "Resolved data directory does not exist. "
            f"Use --data-dir to point at your dataset root. Resolved path: {data_dir}"
        )
    cache_dir = args.output_dir / "prediction_cache"
    metrics_path = metrics_csv_path(args.output_dir)
    if metrics_path.exists() and not args.force_recompute:
        log(f"Loading existing per-sample metrics from {metrics_path}")
        FiHard_table, comusion_table = load_metrics_tables_from_csv(args.output_dir)
    else:
        FiHard_table, _FiHard_payload = evaluate_FiHard(
            checkpoint_path=args.FiHard_checkpoint,
            dataset=args.dataset,
            action_filter=args.action_filter,
            runtime_cfg=runtime_cfg,
            device=device,
            cache_path=cache_dir / "FiHard_predictions.pt",
            force_recompute=bool(args.force_recompute),
        )
        comusion_table, _comusion_payload = evaluate_comusion(
            checkpoint_path=args.comusion_checkpoint,
            dataset=args.dataset,
            action_filter=args.action_filter,
            runtime_cfg=runtime_cfg,
            device=device,
            cache_path=cache_dir / "comusion_predictions.pt",
            force_recompute=bool(args.force_recompute),
        )

    if len(FiHard_table) != len(comusion_table):
        raise RuntimeError(
            f"Sample count mismatch: FiHard={len(FiHard_table)} vs comusion={len(comusion_table)}"
        )

    save_metric_artifacts(
        output_dir=args.output_dir,
        FiHard_table=FiHard_table,
        comusion_table=comusion_table,
        alpha=float(args.alpha),
        rope=float(args.rope),
        rope_mode=str(args.rope_mode),
        nsamples=int(args.nsamples),
    )
    log(f"Wrote Bayesian signed-rank artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
