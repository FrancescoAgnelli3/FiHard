import importlib
import os
import sys
from typing import Dict, Tuple

import torch
import yaml

from .data import create_skeleton

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from common.evaluation import humanmac_metrics_prefixed as _common_humanmac_metrics_prefixed  # type: ignore
from common.evaluation import save_eval_samples_npz  # type: ignore


def _import_splineeqnet_data_modules(splineeqnet_root: str):
    root = os.path.abspath(splineeqnet_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"SplineEqNet root does not exist: {root}")

    if root not in sys.path:
        sys.path.insert(0, root)

    # Ensure we import the sibling repo modules from the requested root.
    for module_name in ("config", "data", "datasets"):
        mod = sys.modules.get(module_name)
        if mod is not None:
            mod_path = os.path.abspath(getattr(mod, "__file__", ""))
            if not mod_path.startswith(root):
                del sys.modules[module_name]

    config_mod = importlib.import_module("config")
    data_mod = importlib.import_module("data")
    return config_mod, data_mod


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


def _select_loader(dataset_split: str, split_strategy: str, train_loader, val_loader, test_loader):
    split = str(dataset_split).lower()
    strategy = str(split_strategy).lower()
    if split in ("train", "training"):
        return train_loader, "train"
    if split in ("valid", "validation", "val"):
        return val_loader, "validation"
    if split == "test":
        if strategy == "test":
            return test_loader, "test"
        return val_loader, "validation"
    raise ValueError(f"Unsupported dataset_split '{dataset_split}' for Assembly evaluation.")


def compute_metrics_assembly(
    dataset_split: str,
    stats_mode: str,
    store_folder: str,
    batch_size: int,
    num_samples: int,
    prepare_model,
    get_prediction,
    process_evaluation_pair,
    **config,
):
    del stats_mode  # kept for compatibility with the default eval path signature

    torch.set_default_dtype(torch.float64 if config["dtype"] == "float64" else torch.float32)

    config_mod, data_mod = _import_splineeqnet_data_modules(config["assembly_splineeqnet_root"])
    DatasetCfg = config_mod.DatasetCfg
    get_dataset_metadata = data_mod.get_dataset_metadata
    build_datasets = data_mod.build_datasets
    make_loaders = data_mod.make_loaders

    dataset_name = str(config.get("assembly_dataset_name", "assembly")).lower()
    metadata = get_dataset_metadata(dataset_name)
    data_dir = config.get("assembly_data_dir") or metadata.get("default_dir", "")
    action_filter = (
        metadata.get("default_action_filter", "")
        if config.get("assembly_action_filter") is None
        else str(config.get("assembly_action_filter"))
    )
    print(
        f"[SplineEqNet eval] dataset='{dataset_name}' | data_dir='{data_dir}' | "
        f"action_filter='{action_filter}'"
    )

    default_wrist_indices = tuple(int(x) for x in metadata.get("default_wrist_indices", (5, 26)))
    requested_wrist_indices = tuple(int(x) for x in config.get("assembly_wrist_indices", default_wrist_indices))
    wrist_indices = requested_wrist_indices if len(requested_wrist_indices) == len(default_wrist_indices) else default_wrist_indices

    ds_cfg = DatasetCfg(
        data_dir=data_dir,
        action_filter=action_filter,
        input_n=int(config["obs_length"]),
        output_n=int(config["pred_length"]),
        stride=int(config.get("assembly_stride", 5)),
        time_interp=config.get("assembly_time_interp"),
        window_norm=config.get("assembly_window_norm"),
        batch_size=int(batch_size),
        eval_batch_mult=int(config.get("assembly_eval_batch_mult", 1)),
        seed=int(config["seed"]),
        wrist_indices=wrist_indices,
        dataset=dataset_name,
        node_count=int(metadata.get("node_count", 21)),
    )

    train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
    train_loader, val_loader, test_loader = make_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=int(ds_cfg.batch_size),
        seed=int(ds_cfg.seed),
        eval_batch_mult=int(ds_cfg.eval_batch_mult),
    )
    eval_loader, eval_split_name = _select_loader(
        dataset_split=dataset_split,
        split_strategy=config.get("assembly_split_strategy", "validation"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    if store_folder is not None:
        store_folder = os.path.join(store_folder, f"obs{ds_cfg.input_n}pred{ds_cfg.output_n}")
        os.makedirs(store_folder, exist_ok=True)

    model_cfg = dict(config)
    if "model_dataset_name_for_loading" in model_cfg and model_cfg["model_dataset_name_for_loading"] is not None:
        model_cfg["dataset_name"] = model_cfg["model_dataset_name_for_loading"]
    skeleton = create_skeleton(**model_cfg)
    model, device, *_ = prepare_model(model_cfg, skeleton)

    num_candidates = max(
        1,
        int(
            config.get(
                "num_candidates",
                config.get("assembly_mpjpe_best_of_k", config.get("humanmac_num_candidates", num_samples)),
            )
        ),
    )
    sample_k = num_candidates
    threshold = float(config.get("humanmac_multimodal_threshold", 0.5))

    total_samples = 0
    best_pred_batches = []
    humanmac_pred_batches = []
    humanmac_tgt_batches = []
    humanmac_context_batches = []

    for batch in eval_loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 3:
            raise RuntimeError("Assembly loader must provide (input, target, norm_factor).")
        inp, out, norm_factor = batch[0], batch[1], batch[2]

        data = inp[..., 4:].to(device).float()
        target = out[..., 4:].to(device).float()
        norm_factor = norm_factor.to(device).float().view(-1)

        with torch.no_grad():
            pred = get_prediction(
                data,
                model,
                num_samples=sample_k,
                pred_length=config["pred_length"],
                diffusion_conditioning=bool(config.get("diffusion_conditioning", True)),
            )
            target_m, pred_m, _, data_m = process_evaluation_pair(
                skeleton=skeleton,
                target=target,
                pred_dict={"pred": pred, "obs": data, "mm_gt": None},
            )

        pred_candidates = pred_m[:, :num_candidates]
        candidate_error = torch.norm(pred_candidates - target_m.unsqueeze(1), dim=-1).mean(dim=(2, 3))
        best_idx = candidate_error.argmin(dim=1)
        best_pred = pred_candidates[torch.arange(pred_candidates.shape[0], device=pred_candidates.device), best_idx]

        batch_size_eff = int(target_m.shape[0])
        total_samples += batch_size_eff

        best_pred_batches.append(best_pred.detach().cpu())
        pred_humanmac = pred_candidates.permute(1, 0, 2, 3, 4).detach().cpu()
        humanmac_pred_batches.append(pred_humanmac)
        humanmac_tgt_batches.append(target_m.detach().cpu())
        humanmac_context_batches.append(data_m.detach().cpu())

    if total_samples == 0:
        raise RuntimeError("Assembly evaluation produced zero samples.")

    humanmac_metrics = _compute_humanmac_metrics(
        pred_candidates=torch.cat(humanmac_pred_batches, dim=1),
        gt_future=torch.cat(humanmac_tgt_batches, dim=0),
        conditioning_context=torch.cat(humanmac_context_batches, dim=0),
        threshold=threshold,
    )
    eval_samples_path = config.get("eval_samples_path")
    if eval_samples_path:
        pred_all_batches = torch.cat(humanmac_pred_batches, dim=1).permute(1, 0, 2, 3, 4).contiguous()
        save_eval_samples_npz(
            eval_samples_path,
            obs=torch.cat(humanmac_context_batches, dim=0),
            target=torch.cat(humanmac_tgt_batches, dim=0),
            pred=torch.cat(best_pred_batches, dim=0),
            pred_all=pred_all_batches,
            metadata={
                "model": str(config.get("method_name", "SkeletonDiffusion")),
                "dataset": dataset_name,
                "action_filter": action_filter,
                "num_candidates": int(num_candidates),
                "eval_split": eval_split_name,
            },
        )

    results = {
        "APD": humanmac_metrics["humanmac_apd"],
        "ADE": humanmac_metrics["humanmac_ade"],
        "FDE": humanmac_metrics["humanmac_fde"],
        "MMADE": humanmac_metrics["humanmac_mmade"],
        "MMFDE": humanmac_metrics["humanmac_mmfde"],
        "CMD": humanmac_metrics["humanmac_cmd"],
        "FID": humanmac_metrics["humanmac_fid"],
        "samples": float(total_samples),
        "assembly_eval_split": eval_split_name,
        "num_candidates": float(num_candidates),
        "humanmac_multimodal_threshold": threshold,
        "test_humanmac_apd_best": humanmac_metrics["humanmac_apd"],
        "test_humanmac_ade_best": humanmac_metrics["humanmac_ade"],
        "test_humanmac_fde_best": humanmac_metrics["humanmac_fde"],
        "test_humanmac_mmade_best": humanmac_metrics["humanmac_mmade"],
        "test_humanmac_mmfde_best": humanmac_metrics["humanmac_mmfde"],
        "test_humanmac_cmd_best": humanmac_metrics["humanmac_cmd"],
        "test_humanmac_fid_best": humanmac_metrics["humanmac_fid"],
        "test_samples": float(total_samples),
        "validation_humanmac_apd_best": humanmac_metrics["humanmac_apd"],
        "validation_humanmac_ade_best": humanmac_metrics["humanmac_ade"],
        "validation_humanmac_fde_best": humanmac_metrics["humanmac_fde"],
        "validation_humanmac_mmade_best": humanmac_metrics["humanmac_mmade"],
        "validation_humanmac_mmfde_best": humanmac_metrics["humanmac_mmfde"],
        "validation_humanmac_cmd_best": humanmac_metrics["humanmac_cmd"],
        "validation_humanmac_fid_best": humanmac_metrics["humanmac_fid"],
        "validation_samples": float(total_samples),
    }

    print("=" * 80)
    print(
        f"Assembly-{eval_split_name}: APD={results['APD']:.6f} | ADE={results['ADE']:.6f} | "
        f"FDE={results['FDE']:.6f} | "
        f"MMADE={results['MMADE']:.6f} | MMFDE={results['MMFDE']:.6f} | "
        f"CMD={results['CMD']:.6f} | FID={results['FID']:.6f} | samples={total_samples}"
    )
    print("=" * 80)

    for key, value in list(results.items()):
        if isinstance(value, torch.Tensor):
            results[key] = float(value.item())

    ov_path = os.path.join(store_folder, f"results_{num_samples}_{dataset_name}.yaml")
    with open(ov_path, "w") as f:
        yaml.dump(results, f, indent=4)
    print(f"Overall results saved to {ov_path}")

    return results
