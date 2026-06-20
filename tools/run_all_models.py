#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
import tempfile
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evaluation import (
    CANONICAL_METRIC_KEYS,
    CANONICAL_LONG_HEADER,
    normalize_metrics_dict,
    read_one_row_csv,
)
from common.preprocessing import default_action_filter

VENDOR = ROOT / "vendor"
PYTHON = os.environ.get("DIFFUSION_HANDS_PYTHON", sys.executable)
DEFAULT_DATA_ROOTS: Dict[str, str] = {
    "assembly": "/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our/",
    "h2o": "/mnt/pve/Turing-Storage2/h2o/",
    "bighands": "/mnt/pve/Turing-Storage2/BigHands/BigHand2.2M/data/",
    "fpha": "/mnt/pve/Turing-Storage2/FPHA/data/",
}
DEFAULT_RUNTIME: Dict[str, str] = {
    "output_root": str(ROOT / "results"),
    "aggregate_csv": str(ROOT / "results" / "all_models_metrics_long.csv"),
}


def _now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _run(cmd: List[str], cwd: Path, env: Optional[dict] = None) -> int:
    print(f"[CMD] ({cwd}) {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(p.returncode)


def _gpu_subprocess_env(cfg: dict) -> dict:
    env = os.environ.copy()
    gpu_index = int(cfg.get("gpu_index", 0))
    if gpu_index >= 0:
        # Restrict each launched backend to a single physical GPU so vendor-specific
        # device selection code cannot accidentally fall back to GPU 0.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def _child_gpu_index(cfg: dict) -> int:
    gpu_index = int(cfg.get("gpu_index", 0))
    # Inside a CUDA_VISIBLE_DEVICES-masked subprocess, the requested physical GPU
    # is exposed as local device 0.
    return 0 if gpu_index >= 0 else gpu_index


def _model_run_root(model_name: str, run_id: str) -> Path:
    run_root = ROOT / "out" / "diffusion_hands_runs" / model_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _maybe_model_run_root(model_name: str, run_id: str, enabled: bool) -> Optional[Path]:
    if not enabled:
        return None
    return _model_run_root(model_name, run_id)


def _vendor_model_run_root(vendor_name: str, model_name: str, run_id: str) -> Path:
    run_root = VENDOR / vendor_name / "out" / "diffusion_hands_runs" / model_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _save_eval_samples_enabled(model_cfg: dict) -> bool:
    return bool(model_cfg.get("save_eval_samples", True))


def _save_model_enabled(cfg: dict) -> bool:
    return bool(cfg.get("save_model", False))


def _checkpoint_dir(model_name: str, run_id: str, cfg: dict) -> Optional[Path]:
    if not _save_model_enabled(cfg):
        return None
    ckpt_dir = _model_run_root(model_name, run_id) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _resolve_model_cfg_entry(model_name: str, entry: object, config_dir: Path) -> dict:
    entry_overrides: Dict[str, object] = {}
    if isinstance(entry, str):
        cfg_path = Path(entry)
        if not cfg_path.is_absolute():
            cfg_path = config_dir / cfg_path
        loaded = _load_yaml(cfg_path)
    elif isinstance(entry, dict) and "config_path" in entry:
        cfg_path = Path(str(entry["config_path"]))
        if not cfg_path.is_absolute():
            cfg_path = config_dir / cfg_path
        loaded = _load_yaml(cfg_path)
        entry_overrides = {k: v for k, v in entry.items() if k != "config_path"}
    elif isinstance(entry, dict):
        # Backward-compatible behavior:
        # - inline config dicts are supported directly
        # - lightweight overrides (e.g. enabled: false) default to
        #   configs/models/{model_name}.yaml
        inline_keys = {"model", "train", "eval", "options", "defaults"}
        if inline_keys.intersection(entry.keys()):
            loaded = dict(entry)
        else:
            cfg_path = config_dir / "models" / f"{model_name}.yaml"
            loaded = _load_yaml(cfg_path)
            entry_overrides = dict(entry)
    else:
        raise ValueError(f"Invalid models.{model_name} entry. Expected path string or mapping, got {type(entry)!r}.")

    if not isinstance(loaded, dict):
        raise ValueError(f"Model config for '{model_name}' must be a mapping.")
    declared = str(loaded.get("model", model_name)).strip().lower()
    base_model = str(entry_overrides.get("base_model", declared)).strip().lower()
    if declared != base_model:
        raise ValueError(
            f"Model config mismatch: entry '{model_name}' declares base_model '{base_model}' "
            f"but loaded config model is '{declared}'."
        )
    if declared != model_name and "base_model" not in entry_overrides:
        raise ValueError(
            f"Model config mismatch: key '{model_name}' vs model '{declared}'. "
            "Use 'base_model' when defining an alias entry."
        )
    if entry_overrides:
        loaded = _deep_merge_dicts(dict(loaded), entry_overrides)
    return {
        "model": model_name,
        "base_model": base_model,
        "enabled": bool(loaded.get("enabled", True)),
        "train": dict(loaded.get("train", {}) or {}),
        "eval": dict(loaded.get("eval", {}) or {}),
        "options": dict(loaded.get("options", {}) or {}),
        "defaults": dict(loaded.get("defaults", {}) or {}),
        "notes": str(loaded.get("notes", "") or ""),
    }


def _resolve_models_config(cfg: dict, config_path: Path) -> dict:
    model_entries = cfg.get("models", {})
    if not isinstance(model_entries, dict):
        raise ValueError("'models' must be a mapping in experiment config.")
    resolved = {}
    config_dir = config_path.parent
    for model_name, entry in model_entries.items():
        resolved[str(model_name).strip().lower()] = _resolve_model_cfg_entry(
            str(model_name).strip().lower(), entry, config_dir
        )
    cfg["models"] = resolved
    return cfg


def _append_long_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = CANONICAL_LONG_HEADER
    if path.exists():
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_header = list(reader.fieldnames or [])
            if existing_header != header:
                existing_rows = list(reader)
                with open(path, "w", newline="", encoding="utf-8") as wf:
                    ww = csv.DictWriter(wf, fieldnames=header)
                    ww.writeheader()
                    for old_row in existing_rows:
                        ww.writerow({k: old_row.get(k, "") for k in header})
    write_header = not path.exists()
    if path.exists() and path.stat().st_size > 0:
        # Guard against previously interrupted writes leaving the file without a trailing newline.
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) not in (b"\n", b"\r"):
                f.write(b"\n")
    serialized: Dict[str, object] = {}
    for k in header:
        v = row.get(k, "")
        if k in {"APD", "ADE", "FDE", "MMADE", "MMFDE", "CMD", "FID"} and isinstance(v, (int, float)):
            serialized[k] = f"{float(v):.3f}"
        else:
            serialized[k] = v
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(serialized)


def _as_dict(obj: object) -> Dict[str, object]:
    return dict(obj) if isinstance(obj, dict) else {}


def _resolve_shared_preprocessing(cfg: dict) -> Dict[str, object]:
    raw = _as_dict(cfg.get("preprocessing"))

    def _int(key: str, default: int) -> int:
        value = raw.get(key, default)
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"preprocessing.{key} must be an integer, got {value!r}")
        if out <= 0:
            raise ValueError(f"preprocessing.{key} must be > 0, got {out}")
        return out

    def _int_or_none(key: str, default: Optional[int]) -> Optional[int]:
        value = raw.get(key, default)
        if value in (None, "", "None", "null"):
            return None
        try:
            out = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"preprocessing.{key} must be an integer or null, got {value!r}")
        if out <= 0:
            raise ValueError(f"preprocessing.{key} must be > 0 when set, got {out}")
        return out

    return {
        "input_n": _int("input_n", 70),
        "output_n": _int("output_n", 30),
        "stride": _int("stride", 5),
        "time_interp": _int_or_none("time_interp", None),
        "window_norm": _int_or_none("window_norm", None),
        "eval_batch_mult": _int("eval_batch_mult", 1),
    }


def _hydra_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        if v == "":
            return "''"
        return v
    return json.dumps(v, separators=(",", ":"))


def _flatten_hydra(prefix: str, obj: Dict[str, object]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for k, v in obj.items():
        if str(k) == "defaults":
            continue
        if isinstance(v, str) and (v == "" or "${" in v):
            # Keep empty/interpolated values in the underlying Hydra config;
            # passing them as CLI overrides is error-prone.
            continue
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_hydra(key, v))
        else:
            out.append((key, _hydra_value(v)))
    return out


def _resolve_data_roots(cfg: dict) -> Dict[str, str]:
    roots = dict(DEFAULT_DATA_ROOTS)
    user_roots = cfg.get("data_roots")
    if user_roots is None:
        return roots
    if not isinstance(user_roots, dict):
        raise ValueError("'data_roots' must be a mapping when provided in experiment config.")
    for k, v in user_roots.items():
        key = str(k).strip().lower()
        if not key:
            continue
        roots[key] = str(v)
    return roots


def _resolve_runtime(cfg: dict) -> Dict[str, str]:
    runtime = dict(DEFAULT_RUNTIME)
    user_runtime = cfg.get("runtime")
    if user_runtime is None:
        return runtime
    if not isinstance(user_runtime, dict):
        raise ValueError("'runtime' must be a mapping when provided in experiment config.")
    for k, v in user_runtime.items():
        runtime[str(k)] = str(v)
    return runtime


def run_card(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "splineeqnet"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    best_cfg = deepcopy(_as_dict(mcfg.get("defaults")))
    if not best_cfg:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    best_cfg["model"] = "card"
    best_cfg["input_n"] = int(pp["input_n"])
    best_cfg["output_n"] = int(pp["output_n"])
    best_cfg["stride"] = int(pp["stride"])
    best_cfg["time_interp"] = pp.get("time_interp")
    best_cfg["window_norm"] = pp.get("window_norm")
    epochs = mcfg.get("train", {}).get("epochs")
    if epochs is not None:
        best_cfg["epochs"] = int(epochs)
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        if "enabled" in es_cfg:
            best_cfg["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            best_cfg["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            best_cfg["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            best_cfg["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            best_cfg["early_stopping_monitor"] = str(es_cfg.get("monitor"))

    eval_examples_path = samples_root / "eval_samples.npz" if samples_root is not None else None
    with tempfile.TemporaryDirectory(prefix="card_") as td:
        temp_root = Path(td)
        run_root = (
            _vendor_model_run_root("splineeqnet", model_name, run_id)
            if save_eval_samples
            else temp_root / "run"
        )
        run_root.mkdir(parents=True, exist_ok=True)
        best_json = run_root / "card_best_config.json"
        out_eval_base = run_root / "card_eval.csv"
        save_root = run_root / "checkpoints"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({"card": best_cfg}, f, indent=2)
        rc = _run(
            [
                PYTHON,
                "train_best_models.py",
                "--model",
                "card",
                "--dataset",
                str(dataset),
                "--data-dir",
                str(data_dir),
                "--action-filter",
                str(action_filter),
                "--eval-split",
                "test",
                "--seed",
                str(cfg["seed"]),
                "--window",
                str(int(pp["input_n"])),
                "--horizon",
                str(int(pp["output_n"])),
                "--eval-batch-mult",
                str(int(pp["eval_batch_mult"])),
                "--best-json",
                str(best_json),
                "--num-candidates",
                str(cfg["num_candidates"]),
                "--humanmac-multimodal-threshold",
                str(cfg["humanmac_multimodal_threshold"]),
                *(["--epochs", str(int(epochs))] if epochs is not None else []),
                *(["--save-eval-examples"] if save_eval_samples else []),
                *(["--eval-examples-path", str(eval_examples_path)] if eval_examples_path is not None else []),
                "--results-csv",
                str(out_eval_base),
                "--save-root",
                str(save_root),
                *(["--final-checkpoint-path", str(checkpoint_dir / "final.pt")] if checkpoint_dir is not None else []),
            ],
            cwd=wd,
            env=_gpu_subprocess_env(cfg),
        )
        if rc != 0:
            raise RuntimeError("card training failed")

        out_candidates = sorted(run_root.glob("**/*.csv"), key=os.path.getmtime)
        if not out_candidates:
            raise RuntimeError(f"card did not produce any metrics CSV under {run_root}")
        row = read_one_row_csv(out_candidates[-1])

    return normalize_metrics_dict(row)


def run_belfusion(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "belfusion"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    run_root = _vendor_model_run_root("belfusion", model_name, run_id)
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")

    template["seed"] = int(cfg["seed"])
    template.setdefault("data", {})
    template["data"]["dataset"] = str(dataset).lower()
    template["data"]["data_dir"] = str(data_dir)
    template["data"]["action_filter"] = str(action_filter)
    template["data"]["input_n"] = int(pp["input_n"])
    template["data"]["output_n"] = int(pp["output_n"])
    template["data"]["stride"] = int(pp["stride"])
    template["data"]["time_interp"] = pp.get("time_interp")
    template["data"]["window_norm"] = pp.get("window_norm")

    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        template.setdefault("train", {})
        template["train"]["epochs"] = int(train_epochs)
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        template.setdefault("train", {})
        if "enabled" in es_cfg:
            template["train"]["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["train"]["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["train"]["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["train"]["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["train"]["early_stopping_monitor"] = str(es_cfg.get("monitor"))

    template.setdefault("eval", {})
    template["eval"]["num_candidates"] = int(cfg["num_candidates"])
    template["eval"]["multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])

    with tempfile.TemporaryDirectory(prefix="belfusion_") as td:
        td_path = Path(td)
        cfg_path = td_path / "belfusion.yaml"
        out_root = run_root
        out_eval = out_root / "eval_stats.csv"
        template.setdefault("runtime", {})
        template["runtime"]["output_dir"] = str(out_root)
        template["runtime"]["metrics_csv"] = str(out_eval)
        if samples_root is not None:
            template["runtime"]["eval_samples_path"] = str(samples_root / "eval_samples.npz")
        _dump_yaml(cfg_path, template)

        cmd = [
            PYTHON,
            "run_belfusion.py",
            "--config",
            str(cfg_path),
            "--seed",
            str(cfg["seed"]),
            "--data-dir",
            str(data_dir),
            "--action-filter",
            str(action_filter),
            "--num-candidates",
            str(cfg["num_candidates"]),
            "--multimodal-threshold",
            str(cfg["humanmac_multimodal_threshold"]),
        ]
        rc = _run(cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("BeLFusion run failed")

        row = read_one_row_csv(out_eval)
        return normalize_metrics_dict(row)


def run_comusion(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "comusion"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    cfg_id = f"dh_{run_id}_comusion"
    cfg_path = wd / "cfg" / f"{cfg_id}.yml"
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    template["t_his"] = int(pp["input_n"])
    template["t_pred"] = int(pp["output_n"])
    template["data_specs"]["dataset"] = str(dataset).lower()
    template["data_specs"]["data_dir"] = str(data_dir)
    template["data_specs"]["action_filter"] = str(action_filter)
    template["data_specs"]["stride"] = int(pp["stride"])
    template["data_specs"]["time_interp"] = pp.get("time_interp")
    template["data_specs"]["window_norm"] = pp.get("window_norm")
    template["data_specs"]["eval_batch_mult"] = int(pp["eval_batch_mult"])
    template["data_specs"]["splineeqnet_root"] = str(VENDOR / "splineeqnet")
    # Keep CoMusion's multimodal grouping consistent with the shared
    # experiment-level evaluation threshold used by the other runners.
    template["data_specs"]["multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])
    template["data_specs"]["humanmac_multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])
    template["eval_sample_num"] = int(cfg["num_candidates"])
    template.setdefault("diff_specs", {})
    if template["diff_specs"].get("div_k") is None:
        template["diff_specs"]["div_k"] = int(cfg["num_candidates"])
    else:
        template["diff_specs"]["div_k"] = int(template["diff_specs"]["div_k"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        tepoch = int(train_epochs)
        template["learn_specs"]["train_epoch"] = tepoch
        template["learn_specs"]["num_epoch_fix_lr"] = min(tepoch, int(template["learn_specs"].get("num_epoch_fix_lr", tepoch)))
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        template.setdefault("learn_specs", {})
        if "enabled" in es_cfg:
            template["learn_specs"]["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["learn_specs"]["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["learn_specs"]["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["learn_specs"]["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["learn_specs"]["early_stopping_monitor"] = str(es_cfg.get("monitor"))
    template["logging_specs"]["model_id"] = cfg_id
    template["logging_specs"]["model_path"] = f"./results/{cfg_id}"
    template["logging_specs"]["final_checkpoint_path"] = str(checkpoint_dir / "final.pt") if checkpoint_dir is not None else ""
    if run_root is not None:
        template["eval_samples_path"] = str(run_root / "eval_samples.npz")
    _dump_yaml(cfg_path, template)

    try:
        rc = _run(
            [
                PYTHON,
                "train.py",
                "--cfg",
                cfg_id,
                "--seed",
                str(cfg["seed"]),
                "--gpu_index",
                str(_child_gpu_index(cfg)),
            ],
            cwd=wd,
            env=_gpu_subprocess_env(cfg),
        )
        if rc != 0:
            raise RuntimeError("CoMusion run failed")

        stats_path = wd / "results" / cfg_id / "results" / "eval_stats.csv"
        row = read_one_row_csv(stats_path)
        return normalize_metrics_dict(row)
    finally:
        try:
            cfg_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            shutil.rmtree(wd / "results" / cfg_id, ignore_errors=True)
        except OSError:
            pass


def run_dlow_cvae(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "dlow"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    cfg_id = f"dh_{run_id}_dlow_cvae"
    cfg_path = wd / "motion_pred" / "cfg" / f"{cfg_id}.yml"
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    template["dataset"] = str(dataset).lower()
    template["data_dir"] = str(data_dir)
    template["action_filter"] = str(action_filter)
    template["t_his"] = int(pp["input_n"])
    template["t_pred"] = int(pp["output_n"])
    template["stride"] = int(pp["stride"])
    template["time_interp"] = pp.get("time_interp")
    template["window_norm"] = pp.get("window_norm")
    template["splineeqnet_root"] = str(VENDOR / "splineeqnet")
    template["seed"] = int(cfg["seed"])
    template["nk"] = int(cfg["num_candidates"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        vae_epoch = int(train_epochs)
        template["num_vae_epoch"] = vae_epoch
        template["num_vae_epoch_fix"] = min(vae_epoch, int(template.get("num_vae_epoch_fix", vae_epoch)))
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        if "enabled" in es_cfg:
            template["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["early_stopping_monitor"] = str(es_cfg.get("monitor"))
    save_interval = mcfg.get("train", {}).get("save_model_interval")
    if save_interval is not None:
        template["save_model_interval"] = int(save_interval)
    else:
        template["save_model_interval"] = 0
    if run_root is not None:
        template["eval_samples_path"] = str(run_root / "eval_samples.npz")
    template["final_checkpoint_path"] = str(checkpoint_dir / "final.pt") if checkpoint_dir is not None else ""
    _dump_yaml(cfg_path, template)

    rc = _run(
        [
            PYTHON,
            "motion_pred/exp_vae.py",
            "--cfg",
            cfg_id,
            "--seed",
            str(cfg["seed"]),
            "--gpu_index",
            str(_child_gpu_index(cfg)),
            "--eval_after_train",
            "--eval_sample_num",
            str(cfg["num_candidates"]),
            "--multimodal_threshold",
            str(cfg["humanmac_multimodal_threshold"]),
        ],
        cwd=wd,
        env=_gpu_subprocess_env(cfg),
    )
    if rc != 0:
        raise RuntimeError("DLow CVAE training failed")

    stats_csv = wd / "results" / cfg_id / "results" / "stats_1.csv"
    out: Dict[str, float] = {}
    with open(stats_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metric = row.get("Metric", "")
            if metric and "vae" in row and row["vae"]:
                try:
                    out[metric] = float(row["vae"])
                except ValueError:
                    pass
    try:
        cfg_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        shutil.rmtree(wd / "results" / cfg_id, ignore_errors=True)
    except OSError:
        pass
    return normalize_metrics_dict(out)


def run_humanmac(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "humanmac"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    run_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    cfg_id = f"dh_{run_id}_humanmac"
    cfg_path = wd / "cfg" / f"{cfg_id}.yml"
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
    template["dataset"] = str(dataset).lower()
    template["data_dir"] = str(data_dir)
    template["action_filter"] = str(action_filter)
    template["t_his"] = int(pp["input_n"])
    template["t_pred"] = int(pp["output_n"])
    template["time_interp"] = pp.get("time_interp")
    template["window_norm"] = int(pp["window_norm"]) if pp.get("window_norm") is not None else int(pp["input_n"])
    template["stride"] = int(pp["stride"])
    template["splineeqnet_root"] = str(VENDOR / "splineeqnet")
    template["num_candidates"] = int(cfg["num_candidates"])
    template["mpjpe_best_of_k"] = int(cfg["num_candidates"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        template["num_epoch"] = int(train_epochs)
    es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if es_cfg:
        if "enabled" in es_cfg:
            template["early_stopping_enabled"] = bool(es_cfg.get("enabled"))
        if "patience" in es_cfg:
            template["early_stopping_patience"] = int(es_cfg.get("patience"))
        if "min_delta" in es_cfg:
            template["early_stopping_min_delta"] = float(es_cfg.get("min_delta"))
        if "warmup" in es_cfg:
            template["early_stopping_warmup"] = int(es_cfg.get("warmup"))
        if "monitor" in es_cfg:
            template["early_stopping_monitor"] = str(es_cfg.get("monitor"))
    _dump_yaml(cfg_path, template)

    num_epoch = int(template.get("num_epoch", 200))
    es_enabled = bool(es_cfg.get("enabled", False)) if es_cfg else False
    es_patience = int(es_cfg.get("patience", 20)) if es_cfg else 20
    es_min_delta = float(es_cfg.get("min_delta", 1e-4)) if es_cfg else 1e-4
    es_warmup = int(es_cfg.get("warmup", 0)) if es_cfg else 0
    es_monitor = str(es_cfg.get("monitor", "train_loss")) if es_cfg else "train_loss"
    validate_last_only = False if (es_enabled and es_monitor.strip().lower() == "val_loss") else True
    rc = _run(
        [
            PYTHON,
            "main_comp.py",
            "--cfg",
            cfg_id,
            "--mode",
            "train",
            "--seed",
            str(cfg["seed"]),
            "--split_seed",
            str(cfg["seed"]),
            "--num_candidates",
            str(cfg["num_candidates"]),
            "--multimodal_threshold",
            str(cfg["humanmac_multimodal_threshold"]),
            "--data_dir",
            str(data_dir),
            "--action_filter",
            str(action_filter),
            *(["--eval_samples_path", str(run_root / "eval_samples.npz")] if run_root is not None else []),
            "--num_epoch",
            str(num_epoch),
            "--save_model_interval",
            "0",
            "--save_metrics_interval",
            "1",
            "--validate_last_epoch_only",
            "false" if not validate_last_only else "true",
            "--early_stopping_enabled",
            "true" if es_enabled else "false",
            "--early_stopping_patience",
            str(es_patience),
            "--early_stopping_min_delta",
            str(es_min_delta),
            "--early_stopping_warmup",
            str(es_warmup),
            "--early_stopping_monitor",
            str(es_monitor),
            *(["--final_checkpoint_path", str(checkpoint_dir / "final.pt")] if checkpoint_dir is not None else []),
        ],
        cwd=wd,
        env=_gpu_subprocess_env(cfg),
    )
    if rc != 0:
        raise RuntimeError("HumanMAC run failed")

    candidates = sorted((wd / "results").glob(f"{cfg_id}_*"), key=os.path.getmtime)
    if not candidates:
        raise RuntimeError("HumanMAC result directory not found")
    latest = candidates[-1]
    stats_latest = latest / "results" / "stats_latest.csv"
    if not stats_latest.exists():
        raise RuntimeError(f"HumanMAC training metrics not found in {latest / 'results'}")

    out: Dict[str, float] = {}
    with open(stats_latest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metric = row.get("Metric", "")
            val = row.get("HumanMAC", "")
            if metric and val:
                try:
                    out[metric] = float(val)
                except ValueError:
                    pass
    try:
        cfg_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        shutil.rmtree(latest, ignore_errors=True)
    except OSError:
        pass
    return normalize_metrics_dict(out)


def run_skeletondiffusion(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "skeletondiffusion"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    checkpoint_dir = _checkpoint_dir(model_name, run_id, cfg)
    run_root = wd / "out" / "diffusion_hands_runs" / f"{model_name}_{run_id}"
    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        train_cfg = mcfg.get("train", {})
        defaults = _as_dict(mcfg.get("defaults"))
        if not defaults:
            raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")
        auto_defaults = _as_dict(defaults.get("train_autoencoder"))
        diff_defaults = _as_dict(defaults.get("train_diffusion"))
        eval_defaults = _as_dict(defaults.get("eval"))
        auto_cfg = train_cfg.get("autoencoder", {}) if isinstance(train_cfg.get("autoencoder", {}), dict) else {}
        diff_cfg = train_cfg.get("diffusion", {}) if isinstance(train_cfg.get("diffusion", {}), dict) else {}
        auto_epochs = auto_cfg.get("epochs")
        auto_iters = auto_cfg.get("iter_per_epoch")
        diff_epochs = diff_cfg.get("epochs")
        diff_iters = diff_cfg.get("iter_per_epoch")
        diff_model = str(diff_cfg.get("model", "skeleton_diffusion"))
        auto_es_cfg = _as_dict(auto_cfg.get("early_stopping"))
        diff_es_cfg = _as_dict(diff_cfg.get("early_stopping"))
        fps = int(_as_dict(auto_defaults.get("dataset")).get("fps", 10))
        hist_sec = float(pp["input_n"]) / float(fps)
        pred_sec = float(pp["output_n"]) / float(fps)
        time_interp_val = _hydra_value(pp.get("time_interp"))
        window_norm_val = _hydra_value(pp.get("window_norm"))
        dataset_lower = str(dataset).lower()
        wrist_indices_by_dataset = {
            "assembly": "[5,26]",
            "h2o": "[0,21]",
            "bighands": "[0]",
            "fpha": "[0]",
        }
        wrist_indices_hydra = wrist_indices_by_dataset.get(dataset_lower, "[5,26]")

        autoenc_out = run_root / "autoencoder"
        diff_out = run_root / "diffusion"

        # Autoencoder (original protocol first stage).
        auto_cmd = [PYTHON, "train_autoencoder.py"]
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("", _as_dict(auto_defaults.get("config")))])
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("task", _as_dict(auto_defaults.get("task")))])
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("dataset", _as_dict(auto_defaults.get("dataset")))])
        auto_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("model", _as_dict(auto_defaults.get("model")))])
        auto_cmd.extend(
            [
                "dataset=assembly",
                "task=assembly_hmp",
                f"dataset.assembly_dataset_name={dataset}",
                f"dataset.assembly_splineeqnet_root={VENDOR / 'splineeqnet'}",
                f"dataset.assembly_data_dir={data_dir}",
                f"dataset.assembly_action_filter={action_filter}",
                f"dataset.assembly_wrist_indices={wrist_indices_hydra}",
                f"task.history_sec={hist_sec}",
                f"task.prediction_horizon_sec={pred_sec}",
                f"dataset.assembly_time_interp={time_interp_val}",
                f"dataset.assembly_window_norm={window_norm_val}",
                f"dataset.data_loader_train.stride={int(pp['stride'])}",
                f"dataset.data_loader_train_eval.stride={int(pp['stride'])}",
                f"dataset.data_loader_valid.stride={int(pp['stride'])}",
                f"output_log_path={autoenc_out}",
                *(["model.num_epochs=%d" % int(auto_epochs)] if auto_epochs is not None else []),
                *(["model.num_iter_perepoch=%d" % int(auto_iters)] if auto_iters is not None else []),
                *(["++model.early_stopping_enabled=%s" % ("true" if bool(auto_es_cfg.get("enabled", False)) else "false")] if auto_es_cfg else []),
                *(["++model.early_stopping_patience=%d" % int(auto_es_cfg.get("patience", 20))] if auto_es_cfg else []),
                *(["++model.early_stopping_min_delta=%s" % float(auto_es_cfg.get("min_delta", 1e-4))] if auto_es_cfg else []),
                *(["++model.early_stopping_warmup=%d" % int(auto_es_cfg.get("warmup", 0))] if auto_es_cfg else []),
                *(["++model.early_stopping_monitor=%s" % str(auto_es_cfg.get("monitor", "train_loss"))] if auto_es_cfg else []),
            ]
        )
        rc = _run(auto_cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("skeletondiffusion autoencoder training failed")

        auto_ckpts = sorted((autoenc_out / "checkpoints").glob("checkpoint_*.pt"), key=os.path.getmtime)
        if not auto_ckpts:
            raise RuntimeError("skeletondiffusion autoencoder checkpoint not found")
        auto_ckpt = auto_ckpts[-1]
        if checkpoint_dir is not None:
            shutil.copy2(auto_ckpt, checkpoint_dir / "autoencoder_final.pt")

        # Diffusion (original protocol second stage).
        diff_cmd = [PYTHON, "train_diffusion.py"]
        diff_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("", _as_dict(diff_defaults.get("config")))])
        diff_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("model", _as_dict(diff_defaults.get("model")))])
        diff_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("cov_matrix", _as_dict(diff_defaults.get("cov_matrix")))])
        diff_cmd.extend(
            [
                f"model={diff_model}",
                f"model.pretrained_autoencoder_path={auto_ckpt}",
                f"output_log_path={diff_out}",
                *(["model.num_epochs=%d" % int(diff_epochs)] if diff_epochs is not None else []),
                *(["model.num_iter_perepoch=%d" % int(diff_iters)] if diff_iters is not None else []),
                *(["++model.early_stopping_enabled=%s" % ("true" if bool(diff_es_cfg.get("enabled", False)) else "false")] if diff_es_cfg else []),
                *(["++model.early_stopping_patience=%d" % int(diff_es_cfg.get("patience", 20))] if diff_es_cfg else []),
                *(["++model.early_stopping_min_delta=%s" % float(diff_es_cfg.get("min_delta", 1e-4))] if diff_es_cfg else []),
                *(["++model.early_stopping_warmup=%d" % int(diff_es_cfg.get("warmup", 0))] if diff_es_cfg else []),
                *(["++model.early_stopping_monitor=%s" % str(diff_es_cfg.get("monitor", "train_loss"))] if diff_es_cfg else []),
            ]
        )
        rc = _run(diff_cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("skeletondiffusion diffusion training failed")

        diff_ckpts = sorted((diff_out / "checkpoints").glob("checkpoint_*.pt"), key=os.path.getmtime)
        if not diff_ckpts:
            raise RuntimeError("skeletondiffusion diffusion checkpoint not found")
        diff_ckpt = diff_ckpts[-1]
        if checkpoint_dir is not None:
            shutil.copy2(diff_ckpt, checkpoint_dir / "diffusion_final.pt")

        # Eval (keeps test split, as requested).
        eval_cmd = [PYTHON, "eval.py"]
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("", _as_dict(eval_defaults.get("config")))])
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("task", _as_dict(eval_defaults.get("task")))])
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("method_specs", _as_dict(eval_defaults.get("method_specs")))])
        eval_cmd.extend([f"{k}={v}" for k, v in _flatten_hydra("dataset", _as_dict(eval_defaults.get("dataset")))])
        eval_cmd.extend(
            [
                "dataset=assembly",
                "task=assembly_hmp",
                "dataset_split=test",
                "assembly_split_strategy=test",
                f"assembly_dataset_name={dataset}",
                f"dataset.assembly_dataset_name={dataset}",
                f"checkpoint_path={diff_ckpt}",
                "if_use_splineeqnet_assembly_pipeline=True",
                f"assembly_splineeqnet_root={VENDOR / 'splineeqnet'}",
                f"dataset.assembly_splineeqnet_root={VENDOR / 'splineeqnet'}",
                f"assembly_data_dir={data_dir}",
                f"dataset.assembly_data_dir={data_dir}",
                f"assembly_action_filter={action_filter}",
                f"dataset.assembly_action_filter={action_filter}",
                f"dataset.assembly_wrist_indices={wrist_indices_hydra}",
                f"task.history_sec={hist_sec}",
                f"task.prediction_horizon_sec={pred_sec}",
                f"obs_length={int(pp['input_n'])}",
                f"pred_length={int(pp['output_n'])}",
                f"assembly_stride={int(pp['stride'])}",
                f"assembly_time_interp={time_interp_val}",
                f"dataset.assembly_time_interp={time_interp_val}",
                f"assembly_window_norm={window_norm_val}",
                f"dataset.assembly_window_norm={window_norm_val}",
                f"assembly_eval_batch_mult={int(pp['eval_batch_mult'])}",
                f"num_samples={cfg['num_candidates']}",
                f"num_candidates={cfg['num_candidates']}",
                f"humanmac_multimodal_threshold={cfg['humanmac_multimodal_threshold']}",
                f"seed={cfg['seed']}",
                *([f"++eval_samples_path={samples_root / 'eval_samples.npz'}"] if samples_root is not None else []),
            ]
        )
        rc = _run(eval_cmd, cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("skeletondiffusion eval failed")

        eval_yamls = sorted(glob.glob(str(diff_out / "**" / f"results_*_{dataset}.yaml"), recursive=True), key=os.path.getmtime)
        if not eval_yamls:
            raise RuntimeError("skeletondiffusion eval yaml not found")
        with open(eval_yamls[-1], "r", encoding="utf-8") as f:
            row = yaml.safe_load(f)
        return normalize_metrics_dict({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
    finally:
        try:
            shutil.rmtree(run_root, ignore_errors=True)
        except OSError:
            pass


def run_gsps(model_name: str, dataset: str, data_dir: Path, action_filter: str, cfg: dict, run_id: str) -> Dict[str, object]:
    wd = VENDOR / "gsps"
    mcfg = cfg.get("models", {}).get(model_name, {})
    save_eval_samples = _save_eval_samples_enabled(mcfg)
    pp = _as_dict(cfg.get("_shared_preprocessing"))
    samples_root = _maybe_model_run_root(model_name, run_id, save_eval_samples)
    run_root = _vendor_model_run_root("gsps", model_name, run_id)
    template = deepcopy(_as_dict(mcfg.get("defaults")))
    if not template:
        raise RuntimeError(f"Missing models.{model_name}.defaults in experiment model YAML.")

    template["seed"] = int(cfg["seed"])
    template["gpu_index"] = int(_child_gpu_index(cfg))
    template["dataset"] = str(dataset).lower()
    template["data_dir"] = str(data_dir)
    template["action_filter"] = str(action_filter)
    template["splineeqnet_root"] = str(VENDOR / "splineeqnet")

    template.setdefault("preprocessing", {})
    template["preprocessing"]["input_n"] = int(pp["input_n"])
    template["preprocessing"]["output_n"] = int(pp["output_n"])
    template["preprocessing"]["stride"] = int(pp["stride"])
    template["preprocessing"]["time_interp"] = pp.get("time_interp")
    template["preprocessing"]["window_norm"] = pp.get("window_norm")
    template["preprocessing"]["eval_batch_mult"] = int(pp["eval_batch_mult"])

    template.setdefault("model", {})
    template["model"]["nk"] = int(cfg["num_candidates"])
    train_epochs = mcfg.get("train", {}).get("epochs")
    if train_epochs is not None:
        template.setdefault("train", {})
        template["train"]["epochs"] = int(train_epochs)
    train_batch = mcfg.get("train", {}).get("batch_size")
    if train_batch is not None:
        template.setdefault("train", {})
        template["train"]["batch_size"] = int(train_batch)
    train_es_cfg = _as_dict(_as_dict(mcfg.get("train")).get("early_stopping"))
    if train_es_cfg:
        template.setdefault("train", {})
        if "enabled" in train_es_cfg:
            template["train"]["early_stopping_enabled"] = bool(train_es_cfg.get("enabled"))
        if "patience" in train_es_cfg:
            template["train"]["early_stopping_patience"] = int(train_es_cfg.get("patience"))
        if "min_delta" in train_es_cfg:
            template["train"]["early_stopping_min_delta"] = float(train_es_cfg.get("min_delta"))
        if "warmup" in train_es_cfg:
            template["train"]["early_stopping_warmup"] = int(train_es_cfg.get("warmup"))
        if "monitor" in train_es_cfg:
            template["train"]["early_stopping_monitor"] = str(train_es_cfg.get("monitor"))
    prior_es_cfg = _as_dict(_as_dict(mcfg.get("prior")).get("early_stopping"))
    if prior_es_cfg:
        template.setdefault("prior", {})
        if "enabled" in prior_es_cfg:
            template["prior"]["early_stopping_enabled"] = bool(prior_es_cfg.get("enabled"))
        if "patience" in prior_es_cfg:
            template["prior"]["early_stopping_patience"] = int(prior_es_cfg.get("patience"))
        if "min_delta" in prior_es_cfg:
            template["prior"]["early_stopping_min_delta"] = float(prior_es_cfg.get("min_delta"))
        if "warmup" in prior_es_cfg:
            template["prior"]["early_stopping_warmup"] = int(prior_es_cfg.get("warmup"))
        if "monitor" in prior_es_cfg:
            template["prior"]["early_stopping_monitor"] = str(prior_es_cfg.get("monitor"))

    template.setdefault("eval", {})
    template["eval"]["multimodal_threshold"] = float(cfg["humanmac_multimodal_threshold"])

    with tempfile.TemporaryDirectory(prefix="gsps_") as td:
        td_path = Path(td)
        cfg_path = td_path / "gsps.yaml"
        out_root = run_root
        out_eval = out_root / "eval_stats.csv"
        template.setdefault("runtime", {})
        template["runtime"]["output_dir"] = str(out_root)
        template["runtime"]["metrics_csv"] = str(out_eval)
        if samples_root is not None:
            template["runtime"]["eval_samples_path"] = str(samples_root / "eval_samples.npz")
        _dump_yaml(cfg_path, template)

        rc = _run([PYTHON, "run_gsps.py", "--config", str(cfg_path)], cwd=wd, env=_gpu_subprocess_env(cfg))
        if rc != 0:
            raise RuntimeError("GSPS run failed")

        row = read_one_row_csv(out_eval)
        return normalize_metrics_dict(row)


def _resolve_datasets(cfg: dict) -> List[str]:
    datasets_raw = cfg.get("datasets")
    if datasets_raw is not None:
        if not isinstance(datasets_raw, list) or not datasets_raw:
            raise ValueError("'datasets' must be a non-empty list when provided.")
        out = [str(d).strip().lower() for d in datasets_raw if str(d).strip()]
    else:
        single = str(cfg.get("dataset", "")).strip().lower()
        if not single:
            raise ValueError("Missing 'dataset' (single value) or 'datasets' (non-empty list).")
        out = [single]

    deduped: List[str] = []
    seen = set()
    for d in out:
        if d not in seen:
            deduped.append(d)
            seen.add(d)
    return deduped


def _resolve_action_filters(cfg: dict, dataset: str) -> List[str]:
    # Apply action_filter only to assembly; keep other datasets unfiltered.
    if dataset != "assembly":
        return [""]

    raw = cfg.get("action_filter", "")
    if isinstance(raw, list):
        if not raw:
            raise ValueError("'action_filter' must be a non-empty list when provided as a list.")
        out = [str(action).strip() for action in raw if str(action).strip()]
        if not out:
            raise ValueError("'action_filter' list must contain at least one non-empty action.")
    else:
        configured_action_filter = "" if raw is None else str(raw).strip()
        out = [default_action_filter(dataset, configured_action_filter)]

    deduped: List[str] = []
    seen = set()
    for action_filter in out:
        if action_filter not in seen:
            deduped.append(action_filter)
            seen.add(action_filter)
    return deduped


def _cleanup_legacy_result_artifacts(out_root: Path) -> None:
    # Remove legacy skeletondiffusion runs that were previously written under
    # diffusion_hands/results by older versions of this orchestrator.
    for p in out_root.glob("skeletondiffusion_*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all diffusion_hands models sequentially.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = _load_yaml(config_path)
    cfg = _resolve_models_config(cfg, config_path)
    cfg["_shared_preprocessing"] = _resolve_shared_preprocessing(cfg)
    cfg.setdefault("humanmac_multimodal_threshold", 0.5)

    data_roots = _resolve_data_roots(cfg)
    datasets = _resolve_datasets(cfg)
    unknown = [d for d in datasets if d not in data_roots]
    if unknown:
        raise ValueError(f"Dataset(s) missing in data_roots: {unknown}")


    runtime = _resolve_runtime(cfg)
    out_root = Path(runtime["output_root"])
    aggregate_csv = Path(runtime["aggregate_csv"])

    out_root.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_result_artifacts(out_root)
    print(f"[PREPROCESSING] shared={cfg['_shared_preprocessing']}")

    runners = {
        "card": run_card,
        "belfusion": run_belfusion,
        "comusion": run_comusion,
        "dlow_cvae": run_dlow_cvae,
        "humanmac": run_humanmac,
        "skeletondiffusion": run_skeletondiffusion,
        "gsps": run_gsps,
    }
    for dataset in datasets:
        data_dir = Path(data_roots[dataset])
        for action_filter in _resolve_action_filters(cfg, dataset):
            run_id = f"{dataset}_{action_filter or 'all'}_{_now()}"

            print(f"[DATASET] starting dataset={dataset} action_filter={action_filter!r} data_dir={data_dir}")

            for model_name, model_cfg in cfg.get("models", {}).items():
                if not bool(model_cfg.get("enabled", False)):
                    continue
                base_model = str(model_cfg.get("base_model", model_name)).strip().lower()
                fn = runners.get(base_model)
                if fn is None:
                    raise ValueError(
                        f"Unsupported base_model '{base_model}' for models.{model_name}. "
                        f"Supported models: {sorted(runners)}"
                    )
                row: Dict[str, object] = {
                    "timestamp": _now(),
                    "dataset": dataset,
                    "action_filter": action_filter,
                    "model": model_name,
                    "status": "ok",
                    "notes": str(model_cfg.get("notes", "") or ""),
                    **{k: float("nan") for k in CANONICAL_METRIC_KEYS},
                }
                try:
                    metrics = fn(model_name, dataset, data_dir, action_filter, cfg, run_id)
                    row.update(metrics)
                    print(f"[{dataset}:{model_name}] metrics={metrics}")
                except Exception as exc:
                    row["status"] = "failed"
                    prefix = str(model_cfg.get("notes", "") or "").strip()
                    row["notes"] = f"{prefix} | {exc}" if prefix else str(exc)
                    print(f"[{dataset}:{model_name}] FAILED: {exc}")
                _append_long_csv(aggregate_csv, row)

    print(f"[DONE] aggregate metrics csv: {aggregate_csv}")


if __name__ == "__main__":
    main()
