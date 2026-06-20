import numpy as np
import os
import torch
import hashlib
from scipy.spatial.transform import Rotation as R
from torch.utils.data._utils.collate import default_collate

from .base_dataset import BaseDataset


def add_noise(tensor, noise_level=0.25, noise_std=0.02):
    shape = tensor.shape
    num_elements = shape[-2]
    
    noise = np.random.randn(*tensor.shape) * noise_std
    mask = np.random.rand(*shape[:-1]) < noise_level

    tensor[mask] += noise[mask]
    return tensor

def custom_collate_for_mmgt(data):
    have_mmgt = 'mm_gt' in data[0][2]
    if have_mmgt:
        mm_gt = [d[2].pop('mm_gt') for d in data]
    obs, pred, extra = default_collate(data)
    if have_mmgt:
        extra['mm_gt'] = mm_gt 

    return obs, pred, extra

class MotionDataset(BaseDataset):
    def __init__(self, split, precomputed_folder, skeleton,
                 obs_length: int, pred_length: int, 
                 segments_path=None,
                 stride=1, augmentation=0, da_mirroring=0.0, da_rotations=0.0,
                 dtype='float32',
                 if_consider_hip=False, if_load_mmgt=False, extended_pred_length: int =None, 
                 if_noisy_obs:bool = False, noise_level: float=0.30, noise_std: float = 0.03, 
                 silent=False, **kwargs):                
                        
        # self.annotations_folder = annotations_folder
        self.precomputed_folder = precomputed_folder
        self.segments_path = segments_path 
        self.split = split
        self.skeleton = skeleton
        self.if_load_mmgt = if_load_mmgt
        self.if_noisy_obs = if_noisy_obs
        self.noise_level = noise_level
        self.noise_std = noise_std
        if not silent:
            print(f"Constructing {type(self).__name__} for split ", self.split)
        assert self.split in  ["valid",'train', 'test'], f"{self.split} not implemented"
        if self.split in  ['test']: 
            assert self.segments_path is not None and self.split in self.segments_path
        else: 
            assert self.segments_path is None or (self.segments_path is not None and self.split in self.segments_path)


        assert da_mirroring >= 0.0 and da_mirroring <= 1.0 and da_rotations >= 0.0 and da_rotations <= 1.0, \
            "Data augmentation strategies must be in [0, 1]"
        self.da_mirroring = da_mirroring
        self.da_rotations = da_rotations
        # self.extended_pred_length = extended_pred_length
        if extended_pred_length is not None:
            assert extended_pred_length > pred_length, "Extended pred length must be greater than pred length"
            assert split in ['test', 'valid'], "Extended pred length is only available for test and valid splits"
            pred_length = extended_pred_length

        self.in_eval = self.in_eval = True if split in ['test','valid'] else False 
        
        super().__init__(precomputed_folder, obs_length, pred_length, augmentation=augmentation, stride=stride, dtype=dtype, if_consider_hip=if_consider_hip, silent=silent)
        
        self.load_mmgt()
        if self.split in  ['test'] and ('if_compute_cmd' in kwargs and kwargs['if_compute_cmd']): 
            self._load_mean_motion()  
            # if self.split == 'valid':
            #     print("Note that we are computing the Cumulative Motion Distribution (CMD) metric with data extracted from test split even if we are performin validation")
        if extended_pred_length is not None:
            self.validate_segments_extended_predlength()
            
        print(f"Constructed {type(self).__name__} for split {self.split} with a total of {len(self.segments)} samples")

    def extract_action_label(self, extra):
        return extra['metadata'][self.metadata_class_idx]
            
    def eval(self):
        self.in_eval = True
    def train(self):
        self.in_eval = False

    def load_mmgt(self, path=None):
        if self.if_load_mmgt:
            if self.if_consider_hip:
                suffix = "_hmp" 
            else: 
                suffix = ""
            if path is None:
                mmgt_path = os.path.join(self.precomputed_folder, f"mmgt_{self.split}{suffix}.txt")
            else: 
                mmgt_path = path.replace('.txt', f"{suffix}.txt")
            assert os.path.exists(mmgt_path), "Multimodal GT file does not exist. Please compute it first."
            super().load_mmgt(mmgt_path)
            assert len(self.mm_indces) == len(self.segments), "Multimodal GT file does not have the same number of samples as the dataset"

    def get_classifier(self, device):
        raise NotImplementedError(f"We don't have a classifier for the {type(self).__name__} dataset") 
    
    def _get_hash_str(self, use_all=False):
        use_all = [str(self.obs_length), str(self.pred_length), str(self.stride), str(self.augmentation)] if use_all else []
        to_hash = "".join(tuple(self.subjects + list(self.actions) + 
                [str(self.drop_root), str(self.use_vel)] + use_all))
        return str(hashlib.sha256(str(to_hash).encode('utf-8')).hexdigest())
    
    # def get_custom_segment(self, subject, action, frame_num):
    #     counter = self.dict_indices[subject][action]
    #     obs, pred = self._get_segment(counter, frame_num, frame_num + self.seg_length - 1)
    #     return obs, pred
    
    def _get_mmgt_idx_(self, extra):
        if self.mm_indces is not None and self.if_load_mmgt:
            mm_gt = self._get_mmgt_for_segment(extra['segment_idx'])
            if self.normalize_data:
                mm_gt = self.normalize(mm_gt)
            extra["mm_gt"] = mm_gt
        else:
            ... #extra["mm_gt"] = -1
        return extra
            
    def data_augmentation(self, obs, pred, extra):
        if self.if_load_mmgt:
            mm_gt = extra["mm_gt"]

        if self.da_mirroring != 0:
            # apply mirroring with probability 0.5
            mirroring_idces = [0, 1] # 2 is not used because the person would be upside down
            for m in mirroring_idces:
                if np.random.rand() < self.da_mirroring:
                    # make a copy of obs, pred
                    obs, pred = obs.copy(), pred.copy()
                    # invert sign of first coordinate at last axis for obs, pred
                    obs[..., m] *= -1
                    pred[..., m] *= -1

                    if self.if_load_mmgt:
                        mm_gt = mm_gt.copy()
                        mm_gt[..., m] *= -1
        
        # extra["non_rotated_obs"] = obs.copy()
        # extra["non_rotated_pred"] = pred.copy()
        if self.da_rotations != 0:
            # apply random rotations with probability 1
            rotation_axes = ['z'] # 'x' and 'y' not used because the person could be upside down
            for a in rotation_axes:
                if np.random.rand() < self.da_rotations:
                    degrees = np.random.randint(0, 360)
                    r = R.from_euler(a, degrees, degrees=True).as_matrix().astype(np.float32)
                    obs = (r @ obs.reshape((-1, 3)).T).T.reshape(obs.shape)
                    pred = (r @ pred.reshape((-1, 3)).T).T.reshape(pred.shape)

                    if self.if_load_mmgt:
                        mm_gt = (r @ mm_gt.reshape((-1, 3)).T).T.reshape(mm_gt.shape)

        if self.if_load_mmgt:
            extra["mm_gt"] = mm_gt
        return obs, pred, extra
    
    def iter_thourgh_seqs(self):
        for seq in self.annotations:
            yield seq
            
    def get_segment_from_dataset(self, idx):
        obs, pred, extra = super().__getitem__(idx) 
        extra['adj'] = self.skeleton.adj_matrix
        return obs, pred, extra
    
    def tranform2inputspace(self, obs, pred, extra):
        data = self.skeleton.tranform_to_input_space(torch.cat([torch.from_numpy(obs), torch.from_numpy(pred)], dim=-3))
        obs, pred = data[..., :obs.shape[-3], :, :], data[..., obs.shape[-3]:, :, :]
        if self.if_load_mmgt:
            extra["mm_gt"] = self.skeleton.tranform_to_input_space(torch.from_numpy(extra["mm_gt"]))

        return obs, pred, extra
    
    def __getitem__(self, idx):
        obs, pred, extra = self.get_segment_from_dataset(idx)
        extra = self._get_mmgt_idx_(extra)
        if self.if_noisy_obs:
            obs[..., 1:, :] = add_noise(obs[..., 1:, :], noise_level=self.noise_level, noise_std=self.noise_std)
        if not self.in_eval:
            obs, pred, extra = self.data_augmentation(obs, pred, extra)
        obs, pred, extra = self.tranform2inputspace(obs, pred, extra)

        return obs, pred, extra