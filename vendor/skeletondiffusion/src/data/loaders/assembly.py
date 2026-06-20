import importlib
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


_DATASET_CACHE: Dict[Tuple, Tuple[Dataset, Dataset, Dataset]] = {}


def _import_splineeqnet_data_modules(splineeqnet_root: str):
    root = os.path.abspath(splineeqnet_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"SplineEqNet root does not exist: {root}")

    if root not in sys.path:
        sys.path.insert(0, root)

    # Ensure we import the requested sibling-repo modules.
    for module_name in ("config", "data", "datasets"):
        mod = sys.modules.get(module_name)
        if mod is not None:
            mod_path = os.path.abspath(getattr(mod, "__file__", ""))
            if not mod_path.startswith(root):
                del sys.modules[module_name]

    config_mod = importlib.import_module("config")
    data_mod = importlib.import_module("data")
    return config_mod, data_mod


def _add_noise(tensor: torch.Tensor, noise_level: float = 0.25, noise_std: float = 0.02):
    noise = torch.randn_like(tensor) * float(noise_std)
    mask = (torch.rand(*tensor.shape[:-1], device=tensor.device) < float(noise_level)).unsqueeze(-1)
    return tensor + noise * mask


def _augment(obs: torch.Tensor, pred: torch.Tensor, da_mirroring: float, da_rotations: float):
    # Mirror around x/y axes as in the existing motion loader.
    if da_mirroring > 0:
        for axis in (0, 1):
            if np.random.rand() < da_mirroring:
                obs = obs.clone()
                pred = pred.clone()
                obs[..., axis] *= -1
                pred[..., axis] *= -1

    # Random rotation around z axis only.
    if da_rotations > 0 and np.random.rand() < da_rotations:
        degrees = np.random.randint(0, 360)
        rot = torch.from_numpy(R.from_euler("z", degrees, degrees=True).as_matrix().astype(np.float32)).to(obs.device)
        obs = (rot @ obs.reshape(-1, 3).T).T.reshape(obs.shape)
        pred = (rot @ pred.reshape(-1, 3).T).T.reshape(pred.shape)
    return obs, pred


def _cache_key(
    seed: int,
    dataset_name: str,
    data_dir: str,
    action_filter: str,
    obs_length: int,
    pred_length: int,
    stride: int,
    time_interp,
    window_norm,
    wrist_indices,
    splineeqnet_root: str,
):
    return (
        int(seed),
        str(dataset_name).lower(),
        os.path.abspath(data_dir),
        str(action_filter),
        int(obs_length),
        int(pred_length),
        int(stride),
        None if time_interp is None else int(time_interp),
        None if window_norm is None else int(window_norm),
        tuple(int(x) for x in wrist_indices),
        os.path.abspath(splineeqnet_root),
    )


class AssemblyDataset(Dataset):
    """
    Assembly loader for SkeletonDiffusion training.
    It uses the same preprocessing and split pipeline as SplineEqNet/data.py.
    """

    dataset_name = "assembly"
    metadata_class_idx = 0
    idx_to_class = ["assembly"]
    class_to_idx = {"assembly": 0}

    def __init__(
        self,
        split,
        skeleton,
        obs_length: int,
        pred_length: int,
        stride=5,
        augmentation=0,
        da_mirroring=0.0,
        da_rotations=0.0,
        if_load_mmgt=False,
        if_noisy_obs: bool = False,
        noise_level: float = 0.30,
        noise_std: float = 0.03,
        seed: int = 0,
        assembly_splineeqnet_root: str = "/home/agnelli/projects/4D_hands_working/SplineEqNet",
        assembly_dataset_name: str = "assembly",
        assembly_data_dir: str = "",
        assembly_action_filter: str = "",
        assembly_wrist_indices=(5, 26),
        assembly_time_interp=None,
        assembly_window_norm=None,
        silent=False,
        **kwargs,
    ):
        del augmentation, kwargs  # Not used; SplineEqNet dataset already provides windows.
        self.split = str(split).lower()
        assert self.split in ("train", "valid", "test")
        self.skeleton = skeleton
        self.obs_length = int(obs_length)
        self.pred_length = int(pred_length)
        self.da_mirroring = float(da_mirroring)
        self.da_rotations = float(da_rotations)
        self.if_noisy_obs = bool(if_noisy_obs)
        self.noise_level = float(noise_level)
        self.noise_std = float(noise_std)
        self.in_eval = self.split in ("valid", "test")
        self.if_load_mmgt = bool(if_load_mmgt)
        if self.if_load_mmgt:
            raise NotImplementedError("AssemblyDataset does not support mm_gt loading.")

        config_mod, data_mod = _import_splineeqnet_data_modules(assembly_splineeqnet_root)
        DatasetCfg = config_mod.DatasetCfg
        get_dataset_metadata = data_mod.get_dataset_metadata
        build_datasets = data_mod.build_datasets

        dataset_name = str(assembly_dataset_name).lower()
        metadata = get_dataset_metadata(dataset_name)
        data_dir = assembly_data_dir or metadata.get("default_dir", "")
        # Keep empty-string override as a valid value (do not fall back to defaults).
        action_filter = (
            metadata.get("default_action_filter", "")
            if assembly_action_filter is None
            else str(assembly_action_filter)
        )
        self.assembly_data_dir = data_dir
        self.assembly_action_filter = action_filter
        default_wrist_indices = tuple(int(x) for x in metadata.get("default_wrist_indices", (5, 26)))
        try:
            wrist_indices_raw = tuple(int(x) for x in assembly_wrist_indices)
        except Exception:
            wrist_indices_raw = default_wrist_indices
        wrist_indices = wrist_indices_raw if len(wrist_indices_raw) == len(default_wrist_indices) else default_wrist_indices
        key = _cache_key(
            seed=seed,
            dataset_name=dataset_name,
            data_dir=data_dir,
            action_filter=action_filter,
            obs_length=self.obs_length,
            pred_length=self.pred_length,
            stride=stride,
            time_interp=assembly_time_interp,
            window_norm=assembly_window_norm,
            wrist_indices=wrist_indices,
            splineeqnet_root=assembly_splineeqnet_root,
        )

        if key not in _DATASET_CACHE:
            ds_cfg = DatasetCfg(
                data_dir=data_dir,
                action_filter=action_filter,
                input_n=self.obs_length,
                output_n=self.pred_length,
                stride=int(stride),
                time_interp=assembly_time_interp,
                window_norm=assembly_window_norm,
                batch_size=1,
                eval_batch_mult=1,
                seed=int(seed),
                wrist_indices=wrist_indices,
                dataset=dataset_name,
                node_count=int(metadata.get("node_count", 21)),
            )
            _DATASET_CACHE[key] = build_datasets(ds_cfg)

        train_dataset, val_dataset, test_dataset = _DATASET_CACHE[key]
        self.dataset = {"train": train_dataset, "valid": val_dataset, "test": test_dataset}[self.split]
        if not silent:
            print(
                f"Constructed AssemblyDataset split='{self.split}' with {len(self.dataset)} samples | "
                f"dataset='{dataset_name}' | data_dir='{self.assembly_data_dir}' | "
                f"action_filter='{self.assembly_action_filter}'"
            )

    def eval(self):
        self.in_eval = True

    def train(self):
        self.in_eval = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        obs7, pred7, norm_factor = self.dataset[idx]
        obs = obs7[..., 4:].to(dtype=torch.float32)
        pred = pred7[..., 4:].to(dtype=torch.float32)

        if self.if_noisy_obs:
            obs = obs.clone()
            obs[1:] = _add_noise(obs[1:], noise_level=self.noise_level, noise_std=self.noise_std)
        if not self.in_eval:
            obs, pred = _augment(obs, pred, da_mirroring=self.da_mirroring, da_rotations=self.da_rotations)

        all_pose = torch.cat([obs, pred], dim=0)  # (T, 21, 3)
        # Add synthetic root at index 0 so Skeleton code can keep dropping root as in hmp.
        root = torch.zeros((all_pose.shape[0], 1, all_pose.shape[-1]), dtype=all_pose.dtype)
        all_pose_with_root = torch.cat([root, all_pose], dim=1)  # (T, 22, 3)
        transformed = self.skeleton.tranform_to_input_space(all_pose_with_root)
        obs_t = transformed[: self.obs_length]
        pred_t = transformed[self.obs_length :]

        return obs_t, pred_t, {
            "sample_idx": idx,
            "clip_idx": 0,
            "init": 0,
            "end": 0,
            "segment_idx": idx,
            "metadata": ("assembly", str(idx)),
            "norm_factor": float(norm_factor.item() if torch.is_tensor(norm_factor) else norm_factor),
            "adj": self.skeleton.adj_matrix,
        }
