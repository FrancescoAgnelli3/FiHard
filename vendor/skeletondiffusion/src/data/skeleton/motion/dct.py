import numpy as np
import torch

from .centerpose import SkeletonCenterPose

def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N) # DCT-II
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m).float()
        idct_m = torch.from_numpy(idct_m).float()
    return dct_m, idct_m

def dct_transform_torch(data, dct_m, dct_n):
    *B, seq_len, features, d = data.shape
    if len(B)==1:
        data =  torch.einsum('dn,bncf->bdcf', dct_m.to(data.device), data)
    else: 
        data =  torch.einsum('dn,ncf->dcf', dct_m.to(data.device), data)
    return data

def reverse_dct_torch(dct_data, idct_m):
    *B, seq_len, j, d = dct_data.shape
    if len(B)==1:
        dct_data = torch.einsum('dn,bncf->bdcf', idct_m.to(dct_data.device), dct_data)
    elif len(B)==2:
        dct_data = torch.einsum('dn,bsncf->bsdcf', idct_m.to(dct_data.device), dct_data)
    else:
        dct_data = torch.einsum('dn,ncf->dcf', idct_m.to(dct_data.device), dct_data)
    return dct_data


class SkeletonDiscreteCosineTransform(SkeletonCenterPose):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.n_pre = self.pred_length
        self.dct_m_fut, self.idct_m_fut = get_dct_matrix(self.n_pre, is_torch=True)
        self.dct_m_past, self.idct_m_past = get_dct_matrix(self.obs_length, is_torch=True)
        
        
    def tranform_to_input_space_pose_only(self, data):
        data = super().tranform_to_input_space_pose_only(data)
        if data.shape[-3] == self.pred_length:
            data_dct = dct_transform_torch(data, self.dct_m_fut, self.n_pre)
        else:
            obs, fut = data[..., :self.obs_length, :, :], data[..., self.obs_length:, :, :]
            obs_dct = dct_transform_torch(obs, self.dct_m_past, self.obs_length)
            fut_dct = dct_transform_torch(fut, self.dct_m_fut, self.n_pre)
            data_dct = torch.cat([obs_dct, fut_dct], dim=-3)        
        return data_dct
    
    def transform_hip_to_input_space(self, data):
        data = super().transform_hip_to_input_space(data)
        hip, pose = data[..., 0:1, :], data[..., 1:, :]
        assert 0, "This function is not implemented"
        hip_dct = torch.matmul(self.dct_m[:self.n_pre], hip)
        kpts = torch.cat([hip_dct, pose], dim=-2)
        return kpts
    
    ##############################  function to tranform back to metric space ##########################################################################
    
    def transform_hip_to_metric_space(self, kpts):
        hip, pose = kpts[..., 0:1, :], kpts[..., 1:, :]
        hip = torch.matmul(self.idct_m[:self.n_pre], hip)       
        kpts = torch.cat([hip, pose], dim=-2) 
        return kpts
    
    def transform_to_metric_space_pose_only(self, kpts):  
        """This is the only funtion that tkaes as input the pose and returns the pose joints"""
        assert kpts.shape[-3] in[self.pred_length, self.obs_length], "The input shape is not correct"
        idct_m = self.idct_m_fut if kpts.shape[-3] == self.pred_length else self.idct_m_past
        pose = reverse_dct_torch(kpts, idct_m)
        return pose