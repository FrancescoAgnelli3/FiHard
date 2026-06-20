from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch

from common.metrics import (
    compute_all_metrics_single,
    humanmac_metrics,
    humanmac_metrics_prefixed,
    splineeqnet_diffusion_batch_eval,
)


CANONICAL_METRIC_KEYS: List[str] = ["APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"]
CANONICAL_LONG_HEADER: List[str] = [
    "timestamp",
    "dataset",
    "action_filter",
    "model",
    "status",
    *CANONICAL_METRIC_KEYS,
    "notes",
]

METRIC_ALIASES = {
    "APD": ["APD", "test_humanmac_apd_best", "validation_humanmac_apd_best"],
    "ADE": ["ADE", "test_humanmac_ade_best", "validation_humanmac_ade_best"],
    "FDE": ["FDE", "test_humanmac_fde_best", "validation_humanmac_fde_best"],
    "MMADE": ["MMADE", "test_humanmac_mmade_best", "validation_humanmac_mmade_best"],
    "MMFDE": ["MMFDE", "test_humanmac_mmfde_best", "validation_humanmac_mmfde_best"],
    "CMD": ["CMD", "cmd", "humanmac_cmd", "test_humanmac_cmd_best", "validation_humanmac_cmd_best"],
    "FID": ["FID", "fid", "humanmac_fid", "test_humanmac_fid_best", "validation_humanmac_fid_best"],
}


def read_one_row_csv(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV has no rows: {path}")
    row = rows[0]
    out: Dict[str, float] = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def normalize_metrics_dict(src: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for target, candidates in METRIC_ALIASES.items():
        for cand in candidates:
            if cand in src:
                out[target] = float(src[cand])
                break
    return out


def _to_numpy_array(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32, copy=False)
    return value


def save_eval_samples_npz(
    path: Path,
    *,
    obs,
    target,
    pred,
    pred_all=None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "obs": _to_numpy_array(obs),
        "target": _to_numpy_array(target),
        "pred": _to_numpy_array(pred),
    }
    pred_all_np = _to_numpy_array(pred_all)
    if pred_all_np is not None:
        payload["pred_all"] = pred_all_np
    if metadata:
        payload["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))

    np.savez_compressed(out_path, **payload)
    return out_path
