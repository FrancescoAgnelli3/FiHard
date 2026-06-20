import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

AuxFeatureType = Union[torch.Tensor, Dict[str, torch.Tensor]]


def _slice_aux_feature(features: AuxFeatureType, idx: int):
    if torch.is_tensor(features):
        return features[idx]
    if isinstance(features, dict):
        out: Dict[str, torch.Tensor] = {}
        for key, tensor in features.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"aux_features[{key}] must be a tensor, got {type(tensor)}")
            out[key] = tensor[idx]
        return out
    raise TypeError(f"Unsupported aux_features type: {type(features)}")


def _normalize_aux_feature(features: AuxFeatureType, expected_len: int) -> AuxFeatureType:
    if torch.is_tensor(features):
        if int(features.shape[0]) != expected_len:
            raise ValueError(f"aux_features length mismatch: expected {expected_len}, got {features.shape[0]}")
        return features.detach().cpu().float().contiguous()
    if isinstance(features, dict):
        normalized: Dict[str, torch.Tensor] = {}
        for key, tensor in features.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"aux_features[{key}] must be a tensor, got {type(tensor)}")
            if int(tensor.shape[0]) != expected_len:
                raise ValueError(
                    f"aux_features[{key}] length mismatch: expected {expected_len}, got {tensor.shape[0]}"
                )
            normalized[key] = tensor.detach().cpu().float().contiguous()
        return normalized
    raise TypeError(f"Unsupported aux_features type: {type(features)}")


class AssemblyDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 5):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "AssemblyDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        if self.aux_features is not None:
            return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(self.aux_features, i)
        return seq[a:b], seq[b:c], norm_factor

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))

class h2oDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 5):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "h2oDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        if self.aux_features is not None:
            return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(self.aux_features, i)
        return seq[a:b], seq[b:c], norm_factor

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))

class BigHandsDataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 5):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "BigHandsDataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        if self.aux_features is not None:
            return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(self.aux_features, i)
        return seq[a:b], seq[b:c], norm_factor

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))

class FPHADataset(Dataset):
    """Sliding-window dataset over multiple (T, N, 7) sequences."""

    def __init__(self, sequences: List[Union[np.ndarray, Tuple[np.ndarray, float]]], input_n: int, output_n: int, stride: int = 5):
        self.sequences: List[torch.Tensor] = []
        self.norm_factors: List[torch.Tensor] = []
        for item in sequences:
            if isinstance(item, tuple) and len(item) == 2:
                seq_arr, factor = item
            else:
                seq_arr, factor = item, 1.0
            self.sequences.append(torch.from_numpy(seq_arr))
            self.norm_factors.append(torch.tensor(float(factor), dtype=torch.float32))
        self.input_n = int(input_n)
        self.output_n = int(output_n)
        self.stride = max(1, int(stride))
        self.aux_features: Optional[AuxFeatureType] = None
        self.index: List[Tuple[int, int]] = []  # (seq_idx, start_t)
        for si, seq in enumerate(self.sequences):
            T = seq.shape[0]
            max_start = T - (self.input_n + self.output_n)
            if max_start <= 0:
                continue
            for st in range(0, max_start, self.stride):
                self.index.append((si, st))

        print(
            "FPHADataset: "
            f"{len(self.sequences)} sequences, {len(self.index)} samples, "
            f"input_n={self.input_n}, output_n={self.output_n}, stride={self.stride}"
        )

    def __len__(self) -> int:  # pragma: no cover - trivial getter
        return len(self.index)

    def __getitem__(self, i: int):
        si, st = self.index[i]
        seq = self.sequences[si]
        a = st
        b = st + self.input_n
        c = b + self.output_n
        norm_factor = self.norm_factors[si].to(dtype=seq.dtype)
        if self.aux_features is not None:
            return seq[a:b], seq[b:c], norm_factor, _slice_aux_feature(self.aux_features, i)
        return seq[a:b], seq[b:c], norm_factor

    def set_aux_features(self, features: Optional[AuxFeatureType]) -> None:
        if features is None:
            self.aux_features = None
            return
        self.aux_features = _normalize_aux_feature(features, len(self))
