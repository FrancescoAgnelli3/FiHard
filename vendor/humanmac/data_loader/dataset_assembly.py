import glob
import importlib.util
import os
import sys
from typing import Optional, Sequence

import numpy as np

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.dataset_graphs import ASSEMBLY_HAND_GROUPS, get_root_first_single_hand_graph


def _parents_from_links(num_nodes, links, root=0):
    # Build a deterministic BFS tree so visualization bones are coherent with edge topology.
    adj = {i: [] for i in range(num_nodes)}
    for a, b in links:
        if 0 <= a < num_nodes and 0 <= b < num_nodes:
            adj[a].append(b)
            adj[b].append(a)
    for i in range(num_nodes):
        adj[i] = sorted(set(adj[i]))

    parents = [-2] * num_nodes
    if not (0 <= root < num_nodes):
        root = 0
    parents[root] = -1
    queue = [root]
    q_idx = 0
    while q_idx < len(queue):
        u = queue[q_idx]
        q_idx += 1
        for v in adj[u]:
            if parents[v] == -2:
                parents[v] = u
                queue.append(v)

    # Fallback for any disconnected node.
    for i in range(num_nodes):
        if parents[i] == -2:
            parents[i] = root if i != root else -1
    return parents


def _ensure_train_test_split(n_total, test_fraction=0.2):
    n_test = max(1, int(round(test_fraction * n_total)))
    n_train = n_total - n_test
    if n_train < 1:
        n_test = max(1, n_test - 1)
        n_train = n_total - n_test
    if n_train < 1 or n_test < 1:
        raise RuntimeError("Unable to split Assembly files into non-empty train/test partitions.")
    return n_train, n_test


def _load_assembly_raw(file_path):
    arr = np.load(file_path)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array in '{file_path}', got shape {arr.shape}")
    # Assembly files are typically (3, T, 2, 21); convert to (T, 42, 3).
    arr = arr.transpose(0, 1, 3, 2).reshape((arr.shape[0], arr.shape[1], arr.shape[2] * arr.shape[3]))
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected xyz coordinates in '{file_path}', got shape {arr.shape}")
    return arr.astype(np.float32)


def _reorder_root_first(hand_xyz, wrist_local):
    if wrist_local == 0:
        return hand_xyz
    order = [wrist_local] + [i for i in range(hand_xyz.shape[1]) if i != wrist_local]
    return hand_xyz[:, order, :]


def _remove_outliers_timewise(coords, k=20.0, max_drop_frac=0.3, max_consecutive=10):
    if coords.ndim != 3 or coords.shape[-1] != 3:
        return coords
    t_len = coords.shape[0]
    if t_len <= 2:
        return coords
    med = np.median(coords, axis=0, keepdims=True)
    mad = np.median(np.abs(coords - med), axis=0, keepdims=True)
    thr = k * (mad + 1e-8)
    dev = np.abs(coords - med)
    is_outlier_comp = dev > thr
    is_outlier_frame = np.any(is_outlier_comp, axis=(1, 2))
    drop_count = int(is_outlier_frame.sum())
    if drop_count / float(t_len) > max_drop_frac:
        return None
    if drop_count > 0:
        arr = is_outlier_frame.astype(np.int32)
        padded = np.r_[0, arr, 0]
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if starts.size > 0:
            max_run = int(np.max(ends - starts))
            if max_run > max_consecutive:
                return None
    kept = coords[~is_outlier_frame]
    if kept.shape[0] < 3:
        return None
    return kept


def _align_and_scale_hand_frame0_wrist_middle(hand, wrist_l, middle_l, eps=1e-8):
    # Mirrors SplineEqNet: scale by frame-0 wrist->middle length; no rotation applied.
    u = hand[0, middle_l, :3] - hand[0, wrist_l, :3]
    bone_len = float(np.linalg.norm(u))
    if not np.isfinite(bone_len) or bone_len < eps:
        return hand.astype(np.float32), 1.0
    hand = (hand / bone_len).astype(np.float32)
    return hand, bone_len


def _time_interp(coords, out_len, interpolation="linear"):
    if out_len is None:
        return coords
    if int(out_len) <= 0 or int(out_len) == coords.shape[0]:
        return coords
    t0 = coords.shape[0]
    t_orig = np.linspace(0.0, 1.0, t0)
    t_new = np.linspace(0.0, 1.0, int(out_len))
    out = np.zeros((int(out_len), coords.shape[1], 3), dtype=np.float64)
    for i in range(coords.shape[1]):
        for k in range(3):
            if interpolation == "linear":
                out[:, i, k] = np.interp(t_new, t_orig, coords[:, i, k])
            else:
                out[:, i, k] = np.interp(t_new, t_orig, coords[:, i, k])
    return out


def _select_active_hand(seq_xyz, window_norm=None, time_interp=None):
    # Mirrors SplineEqNet calculate_embedding_wrist_centered_split_canonical for Assembly.
    centered = seq_xyz.astype(np.float64, copy=True)
    n_nodes = centered.shape[1]
    for group in ASSEMBLY_HAND_GROUPS:
        nodes = [idx for idx in group["nodes"] if idx < n_nodes]
        if not nodes:
            continue
        wrist_idx = int(group["wrist_index"])
        if 0 <= wrist_idx < n_nodes:
            origin = centered[:, wrist_idx, :3]
            centered[:, nodes, :3] -= origin[:, None, :]

    centered_filtered = _remove_outliers_timewise(
        centered, k=20.0, max_drop_frac=0.3, max_consecutive=10
    )
    if centered_filtered is None:
        return None
    centered_filtered = _time_interp(centered_filtered, time_interp, interpolation="linear")

    hand_candidates = []
    for group in ASSEMBLY_HAND_GROUPS:
        nodes_global = [idx for idx in group["nodes"] if idx < n_nodes]
        if not nodes_global:
            continue
        g2l = {g: li for li, g in enumerate(nodes_global)}
        wrist_g = int(group["wrist_index"])
        if wrist_g not in g2l:
            continue
        wrist_l = int(g2l[wrist_g])

        hand = centered_filtered[:, nodes_global, :3].astype(np.float32, copy=True)

        middle_l = 9  # Same local middle MCP assumption as SplineEqNet.
        if 0 <= middle_l < hand.shape[1]:
            hand, norm_factor = _align_and_scale_hand_frame0_wrist_middle(
                hand, wrist_l=wrist_l, middle_l=middle_l, eps=1e-8
            )
        else:
            norm_factor = 1.0

        t_use = hand.shape[0]
        t_win = t_use if window_norm is None or int(window_norm) <= 0 else min(t_use, int(window_norm))
        if t_win < 2:
            continue
        diff = hand[1:t_win] - hand[: t_win - 1]
        motion_energy = float(np.mean(np.linalg.norm(diff, axis=-1)))
        hand_candidates.append((hand.astype(np.float32), float(norm_factor), motion_energy, wrist_l))

    if not hand_candidates:
        return None
    hand_candidates.sort(key=lambda x: x[2], reverse=True)
    best_hand, best_norm, _, best_wrist_l = hand_candidates[0]
    best_hand = _reorder_root_first(best_hand, int(best_wrist_l))
    if not np.isfinite(best_norm) or best_norm <= 1e-8:
        best_norm = 1.0
    return best_hand, float(best_norm)


class DatasetAssembly(Dataset):
    def __init__(
        self,
        mode,
        t_his=25,
        t_pred=100,
        actions='all',
        dataset_name: str = 'assembly',
        splineeqnet_root: str = '/home/agnelli/projects/diffusion_hands/vendor/splineeqnet',
        data_dir=None,
        action_filter='',
        seed=0,
        subset_files: Optional[Sequence[str]] = None,
        time_interp: Optional[int] = None,
        window_norm: Optional[int] = None,
        stride: int = 5,
    ):
        self.dataset_name = str(dataset_name).lower()
        self.splineeqnet_root = splineeqnet_root
        self.data_dir = data_dir
        self.action_filter = action_filter
        self.seed = int(seed)
        self.subset_files = list(subset_files) if subset_files is not None else None
        self.time_interp = int(time_interp) if time_interp is not None else None
        self.window_norm = int(window_norm) if window_norm is not None else int(t_his)
        self.stride = max(1, int(stride))
        super().__init__(mode, t_his, t_pred, actions)

    def prepare_data(self):
        if self.splineeqnet_root not in sys.path:
            sys.path.insert(0, self.splineeqnet_root)
        spline_cfg_path = os.path.join(self.splineeqnet_root, "config.py")
        spline_data_path = os.path.join(self.splineeqnet_root, "data.py")
        cfg_spec = importlib.util.spec_from_file_location("splineeqnet_config", spline_cfg_path)
        data_spec = importlib.util.spec_from_file_location("splineeqnet_data", spline_data_path)
        if cfg_spec is None or cfg_spec.loader is None or data_spec is None or data_spec.loader is None:
            raise RuntimeError(f"Unable to import SplineEqNet modules from '{self.splineeqnet_root}'.")

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

        DatasetCfg = cfg_mod.DatasetCfg
        build_datasets = data_mod.build_datasets
        get_dataset_metadata = data_mod.get_dataset_metadata

        metadata = get_dataset_metadata(self.dataset_name)
        data_dir = self.data_dir or metadata.get("default_dir", "")
        action_filter = (
            metadata.get("default_action_filter", "")
            if self.action_filter is None
            else str(self.action_filter)
        )
        default_wrist_indices = tuple(int(x) for x in metadata.get("default_wrist_indices", (5, 26)))
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
        split_ds = {"train": train_dataset, "test": test_dataset, "val": val_dataset}
        if self.mode not in split_ds:
            raise ValueError(f"Unknown split '{self.mode}'. Expected one of train/val/test.")

        shared_graph = get_root_first_single_hand_graph(self.dataset_name)
        self.skeleton = Skeleton(
            parents=list(shared_graph["parents"]),
            # Use one side list so the whole hand is rendered with one consistent color.
            joints_left=[],
            joints_right=list(range(1, 21)),
        )
        self.kept_joints = np.arange(21)
        self.norm_factors = {}

        seqs = {}
        selected_ds = split_ds[self.mode]
        for seq_idx, seq_tensor in enumerate(getattr(selected_ds, "sequences", [])):
            hand = seq_tensor[..., 4:].detach().cpu().numpy().astype(np.float32)
            if hand.shape[0] < self.t_total:
                continue
            seq_name = f"{self.dataset_name}_{self.mode}_{seq_idx:06d}"
            seqs[seq_name] = hand
            norm_factor = getattr(selected_ds, "norm_factors", [1.0])[seq_idx]
            self.norm_factors[seq_name] = float(norm_factor)

        if not seqs:
            raise RuntimeError(
                "No valid sequences remained after preprocessing/splitting. "
                "Check file shapes and t_his/t_pred values."
            )

        subject_name = f"{self.dataset_name}_{self.mode}"
        self.subjects = [subject_name]
        self.data = {subject_name: seqs}

    def sample_all_action(self):
        dict_s = self.data[self.subjects[0]]
        sample = []
        for action in dict_s.keys():
            seq = dict_s[action]
            max_start = seq.shape[0] - self.t_total
            if max_start < 0:
                continue
            fr_start = 0 if max_start == 0 else np.random.randint(max_start + 1)
            fr_end = fr_start + self.t_total
            sample.append(seq[fr_start:fr_end][None, ...])
        if not sample:
            raise RuntimeError("No Assembly action has enough frames for sample_all_action.")
        return np.concatenate(sample, axis=0)

    def sample(self):
        dict_s = self.data[self.subjects[0]]
        action = np.random.choice(list(dict_s.keys()))
        seq = dict_s[action]
        max_start = seq.shape[0] - self.t_total
        if max_start < 0:
            raise RuntimeError("Sampled Assembly sequence is shorter than t_total.")
        fr_start = 0 if max_start == 0 else np.random.randint(max_start + 1)
        fr_end = fr_start + self.t_total
        traj = seq[fr_start:fr_end]
        return traj[None, ...]

    def sample_iter_action(self, action_category, dataset_type=None):
        dict_s = self.data[self.subjects[0]]
        if action_category not in dict_s:
            raise KeyError(f"Unknown action '{action_category}' for Assembly dataset.")
        seq = dict_s[action_category]
        max_start = seq.shape[0] - self.t_total
        if max_start < 0:
            raise RuntimeError("Requested Assembly action is shorter than t_total.")
        fr_start = 0 if max_start == 0 else np.random.randint(max_start + 1)
        fr_end = fr_start + self.t_total
        return seq[fr_start:fr_end][None, ...]

    def prepare_iter_action(self, dataset_type=None):
        return list(self.data[self.subjects[0]].keys())

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for action, seq in data_s.items():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    norm_factor = np.array([self.norm_factors.get(action, 1.0)], dtype=np.float32)
                    yield traj, norm_factor


class DatasetAssemblyMulti(DatasetAssembly):
    pass
