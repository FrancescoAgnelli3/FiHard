import torch
import os
import math
from torch.utils.data import DataLoader

from .utils.config import init_obj
from .data import create_skeleton
from .data.loaders import custom_collate_for_mmgt
from .config_metrics import get_best_sample_idx as _get_best_sample_idx
from .data import loaders as dataset_type



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


def get_best_sample(pred, target):
    best_idx = _get_best_sample_idx(pred, target)
    batch_idx = torch.arange(pred.shape[0], device=pred.device)
    return pred[batch_idx, best_idx], (batch_idx, best_idx)


def prepare_eval_dataset(config, split, shuffle=False, augmentation=0, da_mirroring=0, da_rotations=0, drop_last=False, num_workers=None, batch_size=None ,dataset=None, stats_mode="probabilistic", **kwargs):
    data_loader_name = f"data_loader_{split}"

    config[data_loader_name]["shuffle"] = shuffle
    config[data_loader_name]["da_mirroring"] = da_mirroring
    config[data_loader_name]["da_rotations"] = da_rotations
    config[data_loader_name]["augmentation"] = augmentation
    config[data_loader_name]["drop_last"] = drop_last
    if "probabilistic" in stats_mode.lower():
        config[data_loader_name]["if_load_mmgt"] = True
    else: 
        config[data_loader_name]["if_load_mmgt"] = False
    if batch_size is not None:
        config["batch_size"] = batch_size
    if num_workers is not None:
        config["num_workers"] = 0
    
    skeleton = create_skeleton(**config)   
    if dataset == None:
        dataset = init_obj(config, 'dataset_type', dataset_type, split=split, skeleton=skeleton, **(config[data_loader_name]))

    data_loader = DataLoader(dataset, pin_memory= True, shuffle=config[data_loader_name]["shuffle"], batch_size=config["batch_size"], 
                                num_workers=config["num_workers"], drop_last=config[data_loader_name]["drop_last"],collate_fn=custom_collate_for_mmgt)    
    assert len(data_loader) >0, "Dataset is too small and last batch was dropped"
    return data_loader, dataset, skeleton


   
def long_term_prediction_best_every50(data, target, extra, get_prediction, process_evaluation_pair, num_samples, config):
    """
    Generate 50 predictions, pick the one closest to GT, take the last frames as past and from it generate 50 predictions, pick the one closest to GT, and so on."""
    new_data = data
    final_pred_data = []
    final_target_data = []
    n_past_frames = data.shape[-3]
    for idx in range(math.ceil(config['long_term_factor'])):
        pred = get_prediction(new_data, extra=extra) # [batch_size, n_samples, seq_length, num_joints, features]
        if idx == math.ceil(config['long_term_factor']) -1 and int(config['long_term_factor']) != config['long_term_factor']:
            pred = pred[..., :int(config['long_term_factor']*config['pred_length'])%config['pred_length'], :, :]
        target_m, pred, mm_gt, data_metric_space = process_evaluation_pair(target=target[..., idx*config['pred_length']:(idx+1)*config['pred_length'], :, :], 
                                                                        pred_dict={'pred': pred, 'obs': new_data})
        if idx == 0:
            data = data_metric_space
        best_pred, indeces_bool = get_best_sample(pred, target_m)
        final_pred_data.append(best_pred)
        final_target_data.append(target_m)
        new_data = pred[indeces_bool][..., -n_past_frames:, :, :] # cut off the first frames

    final_pred_data = torch.cat(final_pred_data, dim=-3)
    pred = final_pred_data.unsqueeze(1).repeat(1, num_samples, 1, 1, 1)
    target = torch.cat(final_target_data, dim=-3)
    return target, pred, mm_gt, data 

def long_term_prediction_best_first50(data, target, extra, get_prediction, process_evaluation_pair, num_samples, config):
    """
    Generate 50 predictions wiht training future time horizont, 
    select the one closest to GT and then propagate that one (most likely countinuation)"""
    new_data = data
    final_pred_data = []
    final_target_data = []
    
    for idx in range(math.ceil(config['long_term_factor'])):
        if idx == 0:
            pred = get_prediction(new_data, num_samples=num_samples, extra=extra) # [batch_size, n_samples, seq_length, num_joints, features]
        else:
            batch_size, num_samples, seq_length, num_joints, features = new_data.shape
            pred = get_prediction(new_data.reshape(batch_size * num_samples, seq_length, num_joints, features), num_samples=1, extra=extra)
            pred = pred.reshape(batch_size, num_samples, config['pred_length'], num_joints, features)
        if idx == math.ceil(config['long_term_factor']) -1 and int(config['long_term_factor']) != config['long_term_factor']:
            pred = pred[..., :int(config['long_term_factor']*config['pred_length'])%config['pred_length'], :, :]
        target_m, pred, mm_gt, data_metric_space = process_evaluation_pair(
            target=target[..., idx*config['pred_length']:(idx+1)*config['pred_length'], :, :],
            pred_dict={'pred': pred, 'obs': new_data}
            )
        if idx == 0:
            data = data_metric_space
        final_pred_data.append(pred)
        final_target_data.append(target_m)
        new_data = pred[..., -data.shape[-3]:, :, :] # cut off the first frames

    final_pred_data = torch.cat(final_pred_data, dim=-3)
    pred = final_pred_data #.unsqueeze(1).repeat(1, num_samples, 1, 1, 1)
    target = torch.cat(final_target_data, dim=-3)
    return target, pred, mm_gt, data
