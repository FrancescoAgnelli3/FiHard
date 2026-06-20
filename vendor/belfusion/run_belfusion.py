#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evaluation import save_eval_samples_npz
from common.metrics import humanmac_metrics, splineeqnet_diffusion_batch_eval

@dataclass
class WindowSample:
    obs: np.ndarray
    fut: np.ndarray
    norm: float


class WindowDataset(Dataset):
    def __init__(self, samples: Sequence[WindowSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        obs = torch.from_numpy(s.obs).float()
        fut = torch.from_numpy(s.fut).float()
        norm = torch.tensor(float(s.norm), dtype=torch.float32)
        return obs, fut, norm


class BelFusionCVAE(nn.Module):
    def __init__(self, obs_dim: int, fut_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.post_enc = nn.Sequential(
            nn.Linear(obs_dim + fut_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, fut_dim),
        )

    def _obs_feat(self, obs_flat: torch.Tensor) -> torch.Tensor:
        return self.obs_enc(obs_flat)

    def forward(self, obs_flat: torch.Tensor, fut_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_obs = self._obs_feat(obs_flat)
        h_post = self.post_enc(torch.cat([obs_flat, fut_flat], dim=1))
        mu = self.mu(h_post)
        logvar = self.logvar(h_post)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.dec(torch.cat([h_obs, z], dim=1))
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, obs_flat: torch.Tensor, num_samples: int) -> torch.Tensor:
        h_obs = self._obs_feat(obs_flat)
        bs = obs_flat.shape[0]
        h_obs_rep = h_obs.unsqueeze(0).expand(num_samples, bs, h_obs.shape[-1])
        z = torch.randn(num_samples, bs, self.mu.out_features, device=obs_flat.device, dtype=obs_flat.dtype)
        dec_in = torch.cat([h_obs_rep, z], dim=-1)
        out = self.dec(dec_in.reshape(num_samples * bs, -1)).reshape(num_samples, bs, -1)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize(text: str) -> str:
    return (text or "all").replace("/", "_").replace(" ", "_")


def _build_windows_from_sequences(
    sequences: Sequence[torch.Tensor],
    norm_factors: Sequence[torch.Tensor | float],
    input_n: int,
    output_n: int,
    stride: int,
) -> List[WindowSample]:
    samples: List[WindowSample] = []
    total = input_n + output_n
    stride = max(1, int(stride))
    for seq_idx, seq_tensor in enumerate(sequences):
        seq = seq_tensor[..., 4:].detach().cpu().numpy().astype(np.float32)
        if seq.shape[0] < total:
            continue
        norm_raw = norm_factors[seq_idx] if seq_idx < len(norm_factors) else 1.0
        norm = float(norm_raw.item()) if isinstance(norm_raw, torch.Tensor) else float(norm_raw)
        for st in range(0, seq.shape[0] - total + 1, stride):
            obs = seq[st : st + input_n]
            fut = seq[st + input_n : st + total]
            samples.append(WindowSample(obs=obs.astype(np.float32), fut=fut.astype(np.float32), norm=float(norm)))
    return samples


def load_split_samples(cfg: Dict[str, object]) -> Tuple[List[WindowSample], List[WindowSample], List[WindowSample]]:
    splineeqnet_root = Path(
        str(cfg.get("data", {}).get("splineeqnet_root", ROOT / "vendor" / "splineeqnet"))
    ).resolve()
    cfg_path = splineeqnet_root / "config.py"
    data_path = splineeqnet_root / "data.py"
    if not cfg_path.exists() or not data_path.exists():
        raise RuntimeError(f"SplineEqNet modules not found under '{splineeqnet_root}'.")

    cfg_spec = importlib.util.spec_from_file_location("splineeqnet_config", cfg_path)
    data_spec = importlib.util.spec_from_file_location("splineeqnet_data", data_path)
    if cfg_spec is None or cfg_spec.loader is None or data_spec is None or data_spec.loader is None:
        raise RuntimeError(f"Unable to import SplineEqNet modules from '{splineeqnet_root}'.")

    splineeqnet_root_str = str(splineeqnet_root)
    added_root = False
    if splineeqnet_root_str not in sys.path:
        sys.path.insert(0, splineeqnet_root_str)
        added_root = True

    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    data_mod = importlib.util.module_from_spec(data_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    prev_config_mod = sys.modules.get("config")
    sys.modules["config"] = cfg_mod
    try:
        data_spec.loader.exec_module(data_mod)
    finally:
        if prev_config_mod is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prev_config_mod
        if added_root and splineeqnet_root_str in sys.path:
            try:
                sys.path.remove(splineeqnet_root_str)
            except ValueError:
                pass

    DatasetCfg = cfg_mod.DatasetCfg
    build_datasets = data_mod.build_datasets
    get_dataset_metadata = data_mod.get_dataset_metadata

    data_cfg = cfg["data"]
    dataset_name = str(data_cfg.get("dataset", "assembly")).lower()
    metadata = get_dataset_metadata(dataset_name)
    data_dir = str(data_cfg.get("data_dir") or metadata.get("default_dir", ""))
    action_filter_raw = data_cfg.get("action_filter")
    action_filter = metadata.get("default_action_filter", "") if action_filter_raw is None else str(action_filter_raw)
    default_wrist_indices = tuple(int(x) for x in metadata.get("default_wrist_indices", (5, 26)))
    ds_cfg = DatasetCfg(
        data_dir=data_dir,
        action_filter=action_filter,
        input_n=int(data_cfg["input_n"]),
        output_n=int(data_cfg["output_n"]),
        stride=int(data_cfg["stride"]),
        time_interp=data_cfg.get("time_interp"),
        window_norm=data_cfg.get("window_norm"),
        batch_size=1,
        eval_batch_mult=1,
        seed=int(cfg["seed"]),
        wrist_indices=default_wrist_indices,
        dataset=dataset_name,
        node_count=int(metadata.get("node_count", 21)),
        edge_index=tuple(metadata.get("edge_index", ())),
        adjacency=tuple(metadata.get("adjacency", ())),
    )
    train_ds, val_ds, test_ds = build_datasets(ds_cfg)

    kwargs = dict(
        input_n=int(data_cfg["input_n"]),
        output_n=int(data_cfg["output_n"]),
        stride=int(data_cfg["stride"]),
    )
    train = _build_windows_from_sequences(
        getattr(train_ds, "sequences", []),
        getattr(train_ds, "norm_factors", []),
        **kwargs,
    )
    val = _build_windows_from_sequences(
        getattr(val_ds, "sequences", []),
        getattr(val_ds, "norm_factors", []),
        **kwargs,
    )
    test = _build_windows_from_sequences(
        getattr(test_ds, "sequences", []),
        getattr(test_ds, "norm_factors", []),
        **kwargs,
    )
    return train, val, test


def make_loader(samples: List[WindowSample], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(WindowDataset(samples), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def evaluate_model(
    model: BelFusionCVAE,
    test_loader: DataLoader,
    num_samples: int,
    threshold: float,
    output_n: int,
    n_nodes: int,
    device: torch.device,
    eval_samples_path: Path | None = None,
    metadata: Dict[str, object] | None = None,
) -> Dict[str, float]:
    model.eval()
    hm_pred_batches: List[torch.Tensor] = []
    hm_gt_batches: List[torch.Tensor] = []
    hm_context_batches: List[torch.Tensor] = []
    saved_obs: List[torch.Tensor] = []
    saved_tgt: List[torch.Tensor] = []
    saved_pred: List[torch.Tensor] = []
    saved_pred_all: List[torch.Tensor] = []

    with torch.no_grad():
        for obs, fut, norm in test_loader:
            obs = obs.to(device)
            fut = fut.to(device)
            norm = norm.to(device)
            bs = obs.shape[0]

            obs_flat = obs.reshape(bs, -1)
            sampled = model.sample(obs_flat, num_samples=num_samples)
            pred = sampled.reshape(num_samples, bs, output_n, n_nodes * 3)
            gt = fut.reshape(bs, output_n, n_nodes * 3)
            conditioning_context = obs

            res = splineeqnet_diffusion_batch_eval(
                pred_candidates=pred,
                gt_future=gt,
                conditioning_context=conditioning_context,
                norm_factor=norm,
                threshold=threshold,
            )
            hm_pred_batches.append(pred.detach().cpu())
            hm_gt_batches.append(gt.detach().cpu())
            hm_context_batches.append(conditioning_context.detach().cpu())

            pred_first_all = pred[:, 0].reshape(num_samples, output_n, n_nodes, 3)
            gt_first = gt[0].reshape(output_n, n_nodes, 3)
            obs_first = conditioning_context[0]
            best_idx = torch.norm(pred_first_all - gt_first.unsqueeze(0), dim=-1).mean(dim=(1, 2)).argmin()
            saved_obs.append(obs_first.detach().cpu())
            saved_tgt.append(gt_first.detach().cpu())
            saved_pred.append(pred_first_all[best_idx].detach().cpu())
            saved_pred_all.append(pred_first_all.detach().cpu())

    if not hm_pred_batches:
        raise RuntimeError("No valid test windows were produced for BeLFusion evaluation.")

    hm = humanmac_metrics(
        pred_candidates=torch.cat(hm_pred_batches, dim=1),
        gt_future=torch.cat(hm_gt_batches, dim=0),
        conditioning_context=torch.cat(hm_context_batches, dim=0),
        threshold=threshold,
    )

    out = {}
    out.update({k: float(hm[k]) for k in ("APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID")})
    if eval_samples_path is not None and saved_obs:
        save_eval_samples_npz(
            eval_samples_path,
            obs=torch.stack(saved_obs, dim=0),
            target=torch.stack(saved_tgt, dim=0),
            pred=torch.stack(saved_pred, dim=0),
            pred_all=torch.stack(saved_pred_all, dim=0),
            metadata=metadata,
        )
    return out


def train(cfg: Dict[str, object]) -> Dict[str, float]:
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples, val_samples, test_samples = load_split_samples(cfg)
    if not train_samples:
        raise RuntimeError("BeLFusion train split has zero windows after preprocessing.")
    if not test_samples:
        raise RuntimeError("BeLFusion test split has zero windows after preprocessing.")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    input_n = int(data_cfg["input_n"])
    output_n = int(data_cfg["output_n"])
    n_nodes = int(data_cfg.get("n_nodes", 21))
    obs_dim = input_n * n_nodes * 3
    fut_dim = output_n * n_nodes * 3

    model = BelFusionCVAE(
        obs_dim=obs_dim,
        fut_dim=fut_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
    ).to(device)

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = make_loader(train_samples, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_samples, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_samples else None
    test_loader = make_loader(test_samples, batch_size=int(eval_cfg.get("batch_size", batch_size)), shuffle=False, num_workers=num_workers)

    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    epochs = int(train_cfg["epochs"])
    kl_weight = float(train_cfg.get("kl_weight", 1e-3))
    l1_weight = float(train_cfg.get("l1_weight", 1.0))
    es_enabled = bool(train_cfg.get("early_stopping_enabled", False))
    es_patience = max(1, int(train_cfg.get("early_stopping_patience", 20)))
    es_min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
    es_warmup = max(0, int(train_cfg.get("early_stopping_warmup", 0)))
    es_monitor = str(train_cfg.get("early_stopping_monitor", "auto")).strip().lower()
    es_best = None
    es_bad_epochs = 0
    es_best_state = None

    for _epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for obs, fut, _norm in train_loader:
            obs = obs.to(device)
            fut = fut.to(device)
            bs = obs.shape[0]
            obs_flat = obs.reshape(bs, -1)
            fut_flat = fut.reshape(bs, -1)

            recon, mu, logvar = model(obs_flat, fut_flat)
            recon_loss = F.mse_loss(recon, fut_flat) + l1_weight * F.l1_loss(recon, fut_flat)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss_sum += float(loss.item())
            train_steps += 1

        train_loss_epoch = train_loss_sum / max(1, train_steps)
        val_loss_epoch = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for obs, fut, _norm in val_loader:
                    obs = obs.to(device)
                    fut = fut.to(device)
                    bs = obs.shape[0]
                    obs_flat = obs.reshape(bs, -1)
                    fut_flat = fut.reshape(bs, -1)
                    recon, mu, logvar = model(obs_flat, fut_flat)
                    recon_loss = F.mse_loss(recon, fut_flat) + l1_weight * F.l1_loss(recon, fut_flat)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    val_loss = recon_loss + kl_weight * kl
                    val_loss_sum += float(val_loss.item())
                    val_steps += 1
            val_loss_epoch = val_loss_sum / max(1, val_steps)

        log_line = f"[BeLFusion][Epoch {_epoch}/{epochs}] train_loss={train_loss_epoch:.6f}"
        if val_loss_epoch is not None:
            log_line += f" val_loss={val_loss_epoch:.6f}"
        print(log_line)

        if es_enabled and _epoch > es_warmup:
            if es_monitor == "val_loss":
                monitored = val_loss_epoch
            elif es_monitor in {"train_loss", "loss"}:
                monitored = train_loss_epoch
            else:
                monitored = val_loss_epoch if val_loss_epoch is not None else train_loss_epoch

            if monitored is not None and np.isfinite(float(monitored)):
                monitored_f = float(monitored)
                improved = es_best is None or monitored_f < (float(es_best) - es_min_delta)
                if improved:
                    es_best = monitored_f
                    es_best_state = copy.deepcopy(model.state_dict())
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                    if es_bad_epochs >= es_patience:
                        print(
                            f"[BeLFusion][EarlyStop] epoch={_epoch} monitor={es_monitor or 'auto'} "
                            f"best={es_best:.6f} current={monitored_f:.6f}"
                        )
                        break

    if es_best_state is not None:
        model.load_state_dict(es_best_state)

    out_dir = Path(str(runtime_cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        num_samples=int(eval_cfg["num_candidates"]),
        threshold=float(eval_cfg["multimodal_threshold"]),
        output_n=output_n,
        n_nodes=n_nodes,
        device=device,
        eval_samples_path=Path(str(runtime_cfg["eval_samples_path"])) if runtime_cfg.get("eval_samples_path") else None,
        metadata={
            "model": "belfusion",
            "dataset": str(data_cfg.get("dataset", "")),
            "action_filter": str(data_cfg.get("action_filter", "")),
            "num_candidates": int(eval_cfg["num_candidates"]),
        },
    )

    out_csv = Path(str(runtime_cfg["metrics_csv"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"])
        w.writeheader()
        w.writerow(metrics)

    meta = {
        "num_train_windows": len(train_samples),
        "num_val_windows": len(val_samples),
        "num_test_windows": len(test_samples),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return metrics


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BeLFusion (vendored) training + eval for diffusion_hands")
    p.add_argument("--config", required=True, help="Path to resolved belfusion YAML config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--action-filter", default=None)
    p.add_argument("--num-candidates", type=int, default=None)
    p.add_argument("--multimodal-threshold", type=float, default=None)
    return p


def main() -> None:
    args = create_parser().parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.data_dir is not None:
        cfg.setdefault("data", {})["data_dir"] = str(args.data_dir)
    if args.action_filter is not None:
        cfg.setdefault("data", {})["action_filter"] = str(args.action_filter)
    if args.num_candidates is not None:
        cfg.setdefault("eval", {})["num_candidates"] = int(args.num_candidates)
    if args.multimodal_threshold is not None:
        cfg.setdefault("eval", {})["multimodal_threshold"] = float(args.multimodal_threshold)

    train(cfg)


if __name__ == "__main__":
    main()
