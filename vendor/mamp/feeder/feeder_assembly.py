import numpy as np
from torch.utils.data import Dataset

from feeder import tools


class Feeder(Dataset):
    """AssemblyHands feeder for MAMP.

    Expected NPZ keys follow MAMP conventions:
      - x_train, y_train
      - x_test, y_test

    Supported x shapes:
      - (N, T, V, C)
      - (N, T, M, V, C)
      - (N, C, T, V, M)
      - (N, T, V*C)  (M=1 inferred)

    Internal storage is (N, C, T, V, M), matching MAMP/NTU feeder output.
    """

    def __init__(
        self,
        data_path,
        label_path=None,
        p_interval=1,
        split="train",
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=True,
        bone=False,
        vel=False,
    ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        if isinstance(p_interval, (int, float)):
            self.p_interval = [float(p_interval)]
        else:
            self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel

        self.load_data()
        if normalization:
            self.get_mean_map()

    def _to_ctvm(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 5:
            # Already N,C,T,V,M
            if arr.shape[1] in (2, 3):
                return arr
            # N,T,M,V,C
            if arr.shape[-1] in (2, 3):
                return arr.transpose(0, 4, 1, 3, 2)
            raise ValueError(f"Unsupported 5D skeleton layout: {arr.shape}")

        if arr.ndim == 4:
            # N,T,V,C -> add person dim M=1
            if arr.shape[-1] in (2, 3):
                arr = arr[:, :, None, :, :]
                return arr.transpose(0, 4, 1, 3, 2)
            raise ValueError(f"Unsupported 4D skeleton layout: {arr.shape}")

        if arr.ndim == 3:
            # N,T,V*C with C assumed 3
            n, t, d = arr.shape
            if d % 3 != 0:
                raise ValueError(f"Cannot infer joints from flattened shape: {arr.shape}")
            v = d // 3
            arr = arr.reshape(n, t, v, 3)
            arr = arr[:, :, None, :, :]
            return arr.transpose(0, 4, 1, 3, 2)

        raise ValueError(f"Unsupported skeleton tensor rank {arr.ndim} with shape {arr.shape}")

    @staticmethod
    def _labels_from_npz(npz_data, split: str, n_samples: int) -> np.ndarray:
        key = "y_train" if split == "train" else "y_test"
        if key not in npz_data:
            return np.zeros((n_samples,), dtype=np.int64)

        y = npz_data[key]
        if y.ndim == 1:
            return y.astype(np.int64)
        if y.ndim == 2:
            # one-hot
            return np.where(y > 0)[1].astype(np.int64)

        raise ValueError(f"Unsupported label shape for {key}: {y.shape}")

    def load_data(self):
        if self.use_mmap:
            npz_data = np.load(self.data_path, mmap_mode="r")
        else:
            npz_data = np.load(self.data_path)

        if self.split == "train":
            x_key = "x_train"
            name_prefix = "train_"
        elif self.split == "test":
            x_key = "x_test"
            name_prefix = "test_"
        else:
            raise NotImplementedError("data split only supports train/test")

        if x_key not in npz_data:
            raise KeyError(f"Missing key '{x_key}' in {self.data_path}")

        data = npz_data[x_key]
        self.data = self._to_ctvm(data)
        self.label = self._labels_from_npz(npz_data, self.split, self.data.shape[0])
        self.sample_name = [name_prefix + str(i) for i in range(len(self.data))]

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def get_mean_map(self):
        data = self.data
        n, c, t, v, m = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((n * t * m, c * v)).std(axis=0).reshape((c, 1, v, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        label = int(self.label[index])

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if self.window_size is not None and int(self.window_size) > 0:
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # bone mode intentionally omitted for AssemblyHands unless explicit hand topology is provided
        return data_numpy, label, index
