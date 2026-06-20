import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import math
import sys
from pathlib import Path
from .fid import FID
from einops import rearrange

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from common.metrics import (  # type: ignore
    apd as _common_apd,
    ade as _common_ade,
    fde as _common_fde,
    mmade as _common_mmade,
    mmfde as _common_mmfde,
)
from common.evaluation import (  # type: ignore
    humanmac_metrics as _common_humanmac_metrics,
    splineeqnet_diffusion_batch_eval as _common_splineeqnet_diffusion_batch_eval,
)

def time_slice(array, t0, t, axis):
    if t == -1:
        return torch.index_select(array, axis, torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32))
    else:
        return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))


def APD(pred, target, *args, t0=0, t=-1):
    return _common_apd(pred, target, *args, t0=t0, t=t)


def ADE(pred, target, *args, t0=0, t=-1):
    return _common_ade(pred, target, *args, t0=t0, t=t)


def FDE(pred, target, *args, t0=0, t=-1):
    return _common_fde(pred, target, *args, t0=t0, t=t)


def MMADE(pred, target, gt_multi, *args, t0=0, t=-1): # memory efficient version
    return _common_mmade(pred, target, gt_multi, *args, t0=t0, t=t)


def MMFDE(pred, target, gt_multi, *args, t0=0, t=-1):
    return _common_mmfde(pred, target, gt_multi, *args, t0=t0, t=t)


def HUMANMAC_METRICS(pred_candidates, gt_future, conditioning_context, threshold=0.5):
    return _common_humanmac_metrics(
        pred_candidates,
        gt_future,
        conditioning_context=conditioning_context,
        threshold=threshold,
    )


def SPLINEEQNET_DIFFUSION_BATCH_EVAL(pred_candidates, gt_future, conditioning_context, norm_factor, threshold=0.5):
    return _common_splineeqnet_diffusion_batch_eval(
        pred_candidates,
        gt_future,
        conditioning_context=conditioning_context,
        norm_factor=norm_factor,
        threshold=threshold,
    )


def APDE(curr_apds, gt_apds):
    """
    input: current batch apds [b, ]
    input: gt apds [b, ]: if zero, ignore
    return: [b, ], none indicating gt apd is 0
    """
    nonzero_idxs = torch.nonzero(gt_apds)
    zero_idxs = torch.nonzero(gt_apds.eq(0))
    ret = abs(curr_apds - gt_apds)
    ret[zero_idxs] = float('nan')
    return ret


def CMD(val_per_frame, val_ref):
    T = len(val_per_frame) + 1
    return np.sum([(T - t) * np.abs(val_per_frame[t-1] - val_ref) for t in range(1, T)])


def CMD_helper(pred, extra, histogram_data, all_obs_classes):
    """
    pred: [b, num_s, t_pred, NC] -> [batch, num_s, t_pred, joint, 3]
    """
    pred_flat = rearrange(pred, '... (n c) -> ... n c', c=3)
    motion = (torch.linalg.norm(pred_flat[:,:,1:] - pred_flat[:,:,:-1], axis=-1)).mean(axis=1).mean(axis=-1)    

    histogram_data.append(motion.cpu().detach().numpy())    
    classes = extra['act'].numpy()
    all_obs_classes.append(classes)
    
    return


def CMD_pose(dataset, histogram_data, all_obs_classes):
    """
    TODO: validate this function
    """
    ret = 0
    obs_classes = np.concatenate(all_obs_classes, axis=0)
    motion_data = np.concatenate(histogram_data, axis=0)
    motion_data_mean = motion_data.mean(axis=0)

    motion_per_class = np.zeros((dataset.num_actions, motion_data.shape[1]))
    # CMD weighted by class
    for i, (name, class_val_ref) in enumerate(zip(dataset.idx_to_class, dataset.mean_motion_per_class)):
        mask = obs_classes == i
        if mask.sum() == 0:
            continue
        motion_data_mean = motion_data[mask].mean(axis=0)
        motion_per_class[i] = motion_data_mean
        ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / obs_classes.shape[0])
    return ret


def FID_helper(pred, gt, classifier_for_fid, all_pred_activations, all_gt_activations, all_pred_classes, all_gt_classes):
    """
    pred: [b, sample_num, t_pred, NC]
    gt: [b, t_pred, NC]
    """
    b, s = pred.shape[0], pred.shape[1]

    pred_ = rearrange(pred, 'b s t d -> (b s) d t') # [bs, nc, t_pred])
    gt_ = rearrange(gt, 'b t d -> b d t') # [b, nc, t_pred]

    pred_activations = classifier_for_fid.get_fid_features(motion_sequence=pred_).cpu().data.numpy()
    gt_activations = classifier_for_fid.get_fid_features(motion_sequence=gt_).cpu().data.numpy()

    all_pred_activations.append(pred_activations)
    all_gt_activations.append(gt_activations)
    
    pred_classes = classifier_for_fid(motion_sequence=pred_.float()).cpu().data.numpy().argmax(axis=1)
    # recover the batch size and samples dimension
    pred_classes = pred_classes.reshape([b, s])
    gt_classes = classifier_for_fid(motion_sequence=gt_.float()).cpu().data.numpy().argmax(axis=1)
    # append to the list
    all_pred_classes.append(pred_classes)
    all_gt_classes.append(gt_classes)

    return


def FID_pose(all_gt_activations, all_pred_activations):
    ret = 0
    ret = FID(np.concatenate(all_gt_activations, axis=0), np.concatenate(all_pred_activations, axis=0))
    return ret
