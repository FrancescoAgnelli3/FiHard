import os
import sys
from functools import partial
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from ignite.metrics import Metric

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.metrics import (  # type: ignore
    ade,
    apd,
    cumulative_motion_distribution_distance,
    fde,
    frechet_motion_distance,
    mmade,
    mmfde,
)


def _to_float_tensor(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _flatten_batch_motion(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
    if x.dim() == 4:
        return x.reshape(x.shape[0], x.shape[1], -1)
    raise ValueError(f"Unsupported motion tensor shape {tuple(x.shape)}")


def get_best_sample_idx(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_t = _to_float_tensor(pred)
    target_t = _to_float_tensor(target, device=pred_t.device)
    if pred_t.dim() <= target_t.dim():
        return torch.zeros((pred_t.shape[0],), dtype=torch.long, device=pred_t.device)
    pred_flat = _flatten_batch_motion(pred_t)
    target_flat = _flatten_batch_motion(target_t).unsqueeze(1)
    dist = torch.linalg.norm(pred_flat - target_flat, dim=-1).mean(dim=-1)
    return dist.argmin(dim=1)


def choose_best_sample(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_t = _to_float_tensor(pred)
    target_t = _to_float_tensor(target, device=pred_t.device)
    if pred_t.dim() <= target_t.dim():
        return pred_t, target_t
    best_idx = get_best_sample_idx(pred_t, target_t)
    batch_idx = torch.arange(pred_t.shape[0], device=pred_t.device)
    best_pred = pred_t[batch_idx, best_idx]
    return best_pred, target_t


def draw_table(results: Dict[str, float]) -> str:
    if not results:
        return ""
    key_w = max(len(str(k)) for k in results)
    header = f"{'Metric'.ljust(key_w)} | Value"
    sep = f"{'-' * key_w}-|-------"
    rows = [header, sep]
    for key, value in results.items():
        try:
            value_str = f"{float(value):.6f}"
        except (TypeError, ValueError):
            value_str = str(value)
        rows.append(f"{str(key).ljust(key_w)} | {value_str}")
    return "\n".join(rows)


def _limb_lengths(pred: torch.Tensor, limbseq: Sequence[Sequence[int]]) -> torch.Tensor:
    pred_t = _to_float_tensor(pred)
    if pred_t.dim() == 5:
        a = torch.stack([pred_t[..., int(i), :] for i, _ in limbseq], dim=-2)
        b = torch.stack([pred_t[..., int(j), :] for _, j in limbseq], dim=-2)
        return torch.linalg.norm(a - b, dim=-1)
    if pred_t.dim() == 4:
        a = torch.stack([pred_t[..., int(i), :] for i, _ in limbseq], dim=-2)
        b = torch.stack([pred_t[..., int(j), :] for _, j in limbseq], dim=-2)
        return torch.linalg.norm(a - b, dim=-1)
    raise ValueError(f"Unsupported pred shape for limb lengths: {tuple(pred_t.shape)}")


def limb_length_variance(pred: torch.Tensor, limbseq: Sequence[Sequence[int]], mode: str = "mean", **kwargs) -> torch.Tensor:
    del kwargs
    lengths = _limb_lengths(pred, limbseq)
    if lengths.numel() == 0:
        return torch.zeros((1,), dtype=torch.float32, device=lengths.device)
    if mode == "mean":
        return lengths.var(dim=-1, unbiased=False).mean(dim=tuple(range(1, lengths.dim())))
    return lengths.var(unbiased=False)


def limb_length_jitter(pred: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    del kwargs
    lengths = _limb_lengths(pred, limbseq)
    if lengths.shape[-2] <= 1:
        return torch.zeros((lengths.shape[0],), dtype=torch.float32, device=lengths.device)
    delta = lengths[..., 1:, :] - lengths[..., :-1, :]
    return delta.abs().mean(dim=tuple(range(1, delta.dim())))


def limb_length_variation_difference_wrtGT(pred: torch.Tensor, target: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    del kwargs
    pred_var = limb_length_variance(pred, limbseq, mode="raw")
    tgt_var = limb_length_variance(target, limbseq, mode="raw")
    return (pred_var - tgt_var).abs().mean(dim=tuple(range(1, pred_var.dim())))


def limb_length_error(pred: torch.Tensor, target: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    del kwargs
    return (_limb_lengths(pred, limbseq) - _limb_lengths(target, limbseq)).abs().mean(dim=tuple(range(1, pred.dim())))


def limb_stretching_normed_rmse(pred: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    del kwargs
    lengths = _limb_lengths(pred, limbseq)
    centered = lengths - lengths.mean(dim=-2, keepdim=True)
    return torch.sqrt((centered.square()).mean(dim=tuple(range(1, centered.dim()))))


def limb_stretching_normed_mean(pred: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    del kwargs
    lengths = _limb_lengths(pred, limbseq)
    centered = lengths - lengths.mean(dim=-2, keepdim=True)
    return centered.abs().mean(dim=tuple(range(1, centered.dim())))


def limb_jitter_normed_rmse(pred: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    return limb_stretching_normed_rmse(pred, limbseq, **kwargs)


def limb_jitter_normed_mean(pred: torch.Tensor, limbseq: Sequence[Sequence[int]], **kwargs) -> torch.Tensor:
    return limb_stretching_normed_mean(pred, limbseq, **kwargs)


def mae(pred: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    del args, kwargs
    pred_t, target_t = choose_best_sample(pred, target)
    return (pred_t - target_t).abs().reshape(pred_t.shape[0], -1).mean(dim=1)


def mpjpe(pred: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    del args, kwargs
    pred_t, target_t = choose_best_sample(pred, target)
    return torch.linalg.norm(pred_t - target_t, dim=-1).mean(dim=(1, 2))


def lat_apd(pred: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    del target, args, kwargs
    return apd(pred=pred, target=torch.empty(0, device=pred.device if isinstance(pred, torch.Tensor) else None))


def cmd(pred: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    del args, kwargs
    vals = []
    pred_t = _to_float_tensor(pred)
    target_t = _to_float_tensor(target, device=pred_t.device)
    for i in range(pred_t.shape[0]):
        vals.append(cumulative_motion_distribution_distance(pred_t[i], target_t[i : i + 1]))
    return torch.tensor(vals, device=pred_t.device, dtype=torch.float32)


class MetricStorer(Metric):
    def __init__(self, output_transform: Callable = lambda x: x, return_op: str = "avg"):
        super().__init__(output_transform=output_transform)
        self.return_op = str(return_op)

    def reset(self) -> None:
        self._values = []

    def update(self, output) -> None:
        value = output
        if value is None:
            return
        value_t = _to_float_tensor(value)
        if value_t.numel() == 0:
            return
        if value_t.dim() == 0:
            value_t = value_t.unsqueeze(0)
        self._values.append(value_t.reshape(-1).cpu())

    def compute(self) -> torch.Tensor:
        if not self._values:
            return torch.tensor(float("nan"))
        stacked = torch.cat(self._values, dim=0)
        finite = stacked[torch.isfinite(stacked)]
        if finite.numel() == 0:
            return torch.tensor(float("nan"))
        if self.return_op == "max":
            return finite.max()
        return finite.mean()


class MeanPerJointPositionError(Metric):
    def __init__(self, output_transform: Callable = lambda x: x):
        super().__init__(output_transform=output_transform)

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, output) -> None:
        pred, target = output
        pred_t = _to_float_tensor(pred)
        target_t = _to_float_tensor(target, device=pred_t.device)
        err = torch.linalg.norm(pred_t - target_t, dim=-1).mean(dim=(1, 2))
        finite = err[torch.isfinite(err)]
        self._sum += float(finite.sum().item())
        self._count += int(finite.numel())

    def compute(self) -> torch.Tensor:
        if self._count == 0:
            return torch.tensor(float("nan"))
        return torch.tensor(self._sum / self._count, dtype=torch.float32)


class FinalDisplacementError(Metric):
    def __init__(self, output_transform: Callable = lambda x: x):
        super().__init__(output_transform=output_transform)

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, output) -> None:
        pred, target = output
        pred_t = _to_float_tensor(pred)
        target_t = _to_float_tensor(target, device=pred_t.device)
        err = torch.linalg.norm(pred_t[:, -1] - target_t[:, -1], dim=-1).mean(dim=1)
        finite = err[torch.isfinite(err)]
        self._sum += float(finite.sum().item())
        self._count += int(finite.numel())

    def compute(self) -> torch.Tensor:
        if self._count == 0:
            return torch.tensor(float("nan"))
        return torch.tensor(self._sum / self._count, dtype=torch.float32)


class MetricStorerAPDE(MetricStorer):
    def __init__(self, mmapd_gt_path: str, output_transform: Callable = lambda x: x):
        del mmapd_gt_path
        super().__init__(output_transform=output_transform)


class MetricStorerFID(Metric):
    def __init__(self, classifier_path: str, output_transform: Callable = lambda x: x, **kwargs):
        del classifier_path, kwargs
        super().__init__(output_transform=output_transform)

    def reset(self) -> None:
        self._pred = []
        self._target = []

    def update(self, output) -> None:
        pred, target = output
        self._pred.append(_to_float_tensor(pred).cpu())
        self._target.append(_to_float_tensor(target).cpu())

    def compute(self) -> torch.Tensor:
        if not self._pred or not self._target:
            return torch.tensor(float("nan"))
        return torch.tensor(
            frechet_motion_distance(torch.cat(self._pred, dim=1), torch.cat(self._target, dim=0)),
            dtype=torch.float32,
        )


def get_classifier(*args, **kwargs):
    del args, kwargs
    return None


def motion_for_cmd(pred: torch.Tensor) -> torch.Tensor:
    return _to_float_tensor(pred)


def resolve_cmd(pred_motion, labels, idx_to_class=None, mean_motion_per_class=None):
    del labels, idx_to_class, mean_motion_per_class
    pred_t = _to_float_tensor(pred_motion)
    if pred_t.dim() == 5:
        pred_t = pred_t.reshape(pred_t.shape[0] * pred_t.shape[1], pred_t.shape[2], pred_t.shape[3], pred_t.shape[4])
    zeros = torch.zeros_like(pred_t[:1])
    return cumulative_motion_distribution_distance(pred_t, zeros)


class CMDMetricStorer(Metric):
    def __init__(self, final_funct: Callable, output_transform: Callable = lambda x: x):
        super().__init__(output_transform=output_transform)
        self.final_funct = final_funct

    def reset(self) -> None:
        self._payload = []

    def update(self, output) -> None:
        self._payload.append(output)

    def compute(self) -> torch.Tensor:
        if not self._payload:
            return torch.tensor(float("nan"))
        pred_all = []
        labels_all = []
        for pred, labels in self._payload:
            pred_all.append(_to_float_tensor(pred).cpu())
            labels_all.append(np.asarray(labels))
        return torch.tensor(self.final_funct(torch.cat(pred_all, dim=0), np.concatenate(labels_all, axis=0)), dtype=torch.float32)


def get_stats_funcs(stats_mode, skeleton, **kwargs):
    limbseq = skeleton.get_limbseq()
    limbseq_mae = limbseq.copy()
    limb_angles_idx = skeleton.limb_angles_idx.copy()
    del limb_angles_idx

    limb_stretching_normed_mean_scaled = lambda *args, **kw: limb_stretching_normed_mean(*args, **kw) * 100
    limb_jitter_normed_mean_scaled = lambda *args, **kw: limb_jitter_normed_mean(*args, **kw) * 100
    limb_stretching_normed_rmse_scaled = lambda *args, **kw: limb_stretching_normed_rmse(*args, **kw) * 100
    limb_jitter_normed_rmse_scaled = lambda *args, **kw: limb_jitter_normed_rmse(*args, **kw) * 100

    assert not kwargs["if_consider_hip"]
    if "deterministic" in stats_mode.lower():
        stats_func = {
            "ADE": ade,
            "FDE": fde,
            "MAE": partial(mae, limbseq=limbseq_mae),
            "APD": apd,
            "StretchMean": partial(limb_stretching_normed_mean_scaled, limbseq=limbseq),
            "JitterMean": partial(limb_jitter_normed_mean_scaled, limbseq=limbseq),
            "StretchRMSE": partial(limb_stretching_normed_rmse_scaled, limbseq=limbseq),
            "JitterRMSE": partial(limb_jitter_normed_rmse_scaled, limbseq=limbseq),
        }
    elif "probabilistic_orig" == stats_mode.lower():
        stats_func = {"APD": apd, "ADE": ade, "FDE": fde, "MMADE": mmade, "MMFDE": mmfde}
    elif "probabilistic" == stats_mode.lower():
        stats_func = {
            "ADE": ade,
            "FDE": fde,
            "MAE": partial(mae, limbseq=limbseq_mae),
            "MMADE": mmade,
            "MMFDE": mmfde,
            "APD": apd,
            "StretchMean": partial(limb_stretching_normed_mean_scaled, limbseq=limbseq),
            "JitterMean": partial(limb_jitter_normed_mean_scaled, limbseq=limbseq),
            "StretchRMSE": partial(limb_stretching_normed_rmse_scaled, limbseq=limbseq),
            "JitterRMSE": partial(limb_jitter_normed_rmse_scaled, limbseq=limbseq),
        }
    else:
        raise NotImplementedError(f"stats_mode not implemented: {stats_mode}")
    return stats_func


def get_apde_storer(**kwargs):
    return partial(MetricStorerAPDE, mmapd_gt_path=os.path.join(kwargs["annotations_folder"], "mmapd_GT.csv")), lambda pred, **kw: apd(pred=pred, target=torch.empty(0))


def get_fid_storer(**kwargs):
    storer = partial(MetricStorerFID, classifier_path=kwargs["precomputed_folder"], **kwargs)
    return storer, lambda pred, target, **kw: (pred, target)


def get_cmd_storer(dataset, if_consider_hip=False, **kwargs):
    assert not if_consider_hip
    cmdstorer = {
        "CMD": (
            partial(
                CMDMetricStorer,
                final_funct=partial(resolve_cmd, idx_to_class=dataset.idx_to_class, mean_motion_per_class=getattr(dataset, "mean_motion_per_class", None)),
            ),
            lambda pred, extra, **kw: (
                motion_for_cmd(pred.clone()),
                np.array([dataset.class_to_idx[c] for c in extra["metadata"][dataset.metadata_class_idx]]),
            ),
        ),
    }
    return cmdstorer


def attach_engine_to_metrics(engine, dataset_split, stats_mode, dataset, skeleton, if_compute_cmd=False, if_compute_fid=False, if_compute_apde=False, **config):
    stats_func = get_stats_funcs(stats_mode, skeleton=skeleton, **config)

    stats_metrics = {
        k: MetricStorer(output_transform=lambda data, funct=funct: funct(**(data.copy())), return_op="max" if "_max" in k else "avg")
        for k, funct in stats_func.items()
    }

    for name, metric in stats_metrics.items():
        metric.attach(engine, name)

    if dataset_split == "test" and if_compute_fid:
        fid_storer, funct = get_fid_storer(**config)
        apde_metrics = {"FID": fid_storer(output_transform=funct)}
        for name, metric in apde_metrics.items():
            metric.attach(engine, name)

    if dataset_split == "test" and if_compute_cmd:
        cmd_storer = get_cmd_storer(dataset, **config)
        cmd_metrics = {cmd_name: cmdclass(output_transform=lambda x_dict, funct=funct: funct(**x_dict.copy())) for cmd_name, (cmdclass, funct) in cmd_storer.items()}
        for name, metric in cmd_metrics.items():
            metric.attach(engine, name)

    if if_compute_apde:
        storer, funct = get_apde_storer(**config)
        apde_metrics = {"APDE": storer(output_transform=lambda x_dict: funct(**x_dict.copy()))}
        for name, metric in apde_metrics.items():
            metric.attach(engine, name)
