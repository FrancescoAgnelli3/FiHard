#!/usr/bin/env python3
"""Extract global MAMP features from Assembly feeder windows.

Output:
  npz with
    - feats: (N, D) global pooled feature per sample window
    - labels: (N,)
    - indices: (N,)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feeder.feeder_assembly import Feeder
from model_mamp.transformer import Transformer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract global MAMP features for diffusion conditioning.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained MAMP checkpoint (*.pth)")
    p.add_argument("--data-path", type=str, required=True, help="Assembly NPZ generated for MAMP")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--config", type=str, default=None, help="Optional MAMP pretrain config YAML to load model_args")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--mask-ratio", type=float, default=0.0, help="Set >0 to apply MAMP masking during feature extraction")
    p.add_argument("--motion-aware-tau", type=float, default=0.80)

    # fallback model args if --config is not provided
    p.add_argument("--dim-feat", type=int, default=256)
    p.add_argument("--decoder-dim-feat", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--decoder-depth", type=int, default=5)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-frames", type=int, default=70)
    p.add_argument("--num-joints", type=int, default=21)
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--t-patch-size", type=int, default=1)

    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def _model_args_from_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model_args = cfg.get("model_args", None)
    if not isinstance(model_args, dict):
        raise ValueError(f"No model_args dictionary in {config_path}")
    return model_args


def _build_model(args: argparse.Namespace) -> Transformer:
    if args.config:
        model_args = _model_args_from_config(args.config)
    else:
        model_args = {
            "dim_in": 3,
            "dim_feat": int(args.dim_feat),
            "decoder_dim_feat": int(args.decoder_dim_feat),
            "depth": int(args.depth),
            "decoder_depth": int(args.decoder_depth),
            "num_heads": int(args.num_heads),
            "mlp_ratio": float(args.mlp_ratio),
            "num_frames": int(args.num_frames),
            "num_joints": int(args.num_joints),
            "patch_size": int(args.patch_size),
            "t_patch_size": int(args.t_patch_size),
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "norm_skes_loss": True,
        }
    return Transformer(**model_args)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Older checkpoints may include objects like argparse.Namespace.
        # Fallback is acceptable here because the checkpoint is local/trusted.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys: {len(unexpected)}")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))

    device = torch.device(args.device)

    dataset = Feeder(
        data_path=args.data_path,
        split=args.split,
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=-1,
        normalization=False,
        use_mmap=True,
        bone=False,
        vel=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    model = _build_model(args).to(device)
    model.eval()
    _load_checkpoint(model, args.checkpoint, device)

    all_feats = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for samples, labels, indices in loader:
            # samples: (B, C, T, V, M)
            samples = samples.float().to(device, non_blocking=True)
            b, c, t, v, m = samples.shape
            x = samples.permute(0, 4, 2, 3, 1).contiguous().view(b * m, t, v, c)

            latent, _mask, _ids_restore = model.forward_encoder(
                x,
                mask_ratio=float(args.mask_ratio),
                motion_aware_tau=float(args.motion_aware_tau),
            )

            # Global conditioning vector for diffusion stage.
            feat = latent.mean(dim=1).view(b, m, -1).mean(dim=1)

            all_feats.append(feat.cpu())
            all_labels.append(labels.cpu())
            all_indices.append(indices.cpu())

    feats = torch.cat(all_feats, dim=0).numpy().astype(np.float32, copy=False)
    labels = torch.cat(all_labels, dim=0).numpy().astype(np.int64, copy=False)
    indices = torch.cat(all_indices, dim=0).numpy().astype(np.int64, copy=False)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, feats=feats, labels=labels, indices=indices)

    print(f"Saved features to {out_path}")
    print(f"feats shape: {feats.shape}")


if __name__ == "__main__":
    main()
