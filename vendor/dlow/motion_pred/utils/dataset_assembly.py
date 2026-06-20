import sys
from pathlib import Path

import numpy as np

from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.dataset_graphs import get_root_first_single_hand_graph


class DatasetAssembly(Dataset):
    def __init__(
        self,
        mode,
        t_his=70,
        t_pred=30,
        actions="all",
        dataset_name="assembly",
        data_dir=None,
        action_filter="",
        stride=5,
        seed=0,
        time_interp=None,
        window_norm=None,
        splineeqnet_root=None,
        use_vel=False,
        **kwargs,
    ):
        self.dataset_name = str(dataset_name).lower()
        self.data_dir = data_dir
        self.action_filter = action_filter
        self.stride = max(1, int(stride))
        self.seed = int(seed)
        self.time_interp = time_interp
        self.window_norm = int(window_norm) if window_norm is not None else int(t_his)
        self.splineeqnet_root = str(splineeqnet_root or (_PROJECT_ROOT / "vendor" / "splineeqnet"))
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        self.subjects_split = {}
        self.subjects = [self.mode]
        self.kept_joints = np.arange(21)
        shared_graph = get_root_first_single_hand_graph(self.dataset_name)
        self.skeleton = Skeleton(
            parents=list(shared_graph["parents"]),
            joints_left=[],
            joints_right=[],
        )
        self.process_data()

    def process_data(self):
        if self.splineeqnet_root not in sys.path:
            sys.path.insert(0, self.splineeqnet_root)
        from config import DatasetCfg
        from data import build_datasets, get_dataset_metadata

        metadata = get_dataset_metadata(self.dataset_name)
        data_dir = self.data_dir or metadata.get("default_dir", "")
        action_filter = (
            metadata.get("default_action_filter", "")
            if self.action_filter is None
            else str(self.action_filter)
        )
        default_wrist_indices = tuple(int(idx) for idx in metadata.get("default_wrist_indices", (5, 26)))
        ds_cfg = DatasetCfg(
            data_dir=data_dir,
            action_filter=action_filter,
            input_n=int(self.t_his),
            output_n=int(self.t_pred),
            stride=int(self.stride),
            time_interp=self.time_interp,
            window_norm=self.window_norm,
            batch_size=1,
            eval_batch_mult=1,
            seed=int(self.seed),
            wrist_indices=default_wrist_indices,
            dataset=self.dataset_name,
            node_count=int(metadata.get("node_count", 21)),
            edge_index=tuple(metadata.get("edge_index", ())),
            adjacency=tuple(metadata.get("adjacency", ())),
        )
        train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
        split_ds = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
        if self.mode not in split_ds:
            raise ValueError(f"Unknown Assembly split '{self.mode}'. Expected one of train/val/test.")

        sequences = {}
        self.norm_factors = {}
        selected_ds = split_ds[self.mode]
        for seq_idx, seq_tensor in enumerate(getattr(selected_ds, "sequences", [])):
            seq = seq_tensor[..., 4:].detach().cpu().numpy().astype(np.float32)
            if seq.shape[0] < self.t_total:
                continue

            if self.use_vel:
                velocity = np.diff(seq[:, :1], axis=0)
                velocity = np.append(velocity, velocity[[-1]], axis=0)
            seq = seq.copy()
            seq[:, 1:] -= seq[:, :1]  # ensure wrist-relative coordinates like legacy pipeline
            if self.use_vel:
                seq = np.concatenate((seq, velocity), axis=1)
            seq_name = f"{self.dataset_name}_{self.mode}_{seq_idx:06d}"
            sequences[seq_name] = seq.astype(np.float32)
            norm_factor = getattr(selected_ds, "norm_factors", [1.0])[seq_idx]
            self.norm_factors[seq_name] = float(norm_factor)

        if not sequences:
            raise RuntimeError(f"No valid Assembly sequences remained for split '{self.mode}'.")
        self.data = {self.mode: sequences}

    def iter_generator(self, step=None):
        for data_s in self.data.values():
            for seq in data_s.values():
                max_start = seq.shape[0] - self.t_total
                if max_start < 0:
                    continue
                for start in range(0, max_start + 1, self.stride):
                    yield seq[None, start : start + self.t_total]

    def iter_generator_with_scale(self):
        for data_s in self.data.values():
            for seq_name, seq in data_s.items():
                max_start = seq.shape[0] - self.t_total
                if max_start < 0:
                    continue
                norm_factor = self.norm_factors.get(seq_name, 1.0)
                for start in range(0, max_start + 1, self.stride):
                    yield seq[None, start : start + self.t_total], float(norm_factor)
