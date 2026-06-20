import os
import yaml
from hydra import initialize, compose
from omegaconf import OmegaConf

import src.core as model_arch
from .utils.config import init_obj
from .utils.config import load_and_merge_autoenc_cfg

def create_model(device, **kwargs,):
    model = init_obj(kwargs, 'arch', model_arch).to(device)
    return model


def load_model_config_exp(checkpoint_path):
    exp_folder = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(exp_folder, 'config.yaml')
    with open(config_path, 'r') as stream:
        cfg_diffusion = yaml.safe_load(stream)
    cfg = load_and_merge_autoenc_cfg(cfg_diffusion)
    return cfg, exp_folder

def quick_cfg_for_inference_no_hydra(checkpoint_path, dataset_name, num_samples=50):
    with initialize(config_path="../configs/config_eval", version_base="1.3.2"):
        cfg=compose(
        config_name="config.yaml", 
        overrides=[
            f"dataset={dataset_name}",
            "task=hmp",
            "dataset_split=valid",
            "stats_mode=deterministic",
            f"num_samples={num_samples}",
            f"batch_size=1",
            f"checkpoint_path={checkpoint_path}",
        ]
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # for backwards compatibility with the code
    for subconf in ['task', 'dataset']:
        if subconf in cfg:
            cfg = {**cfg, **cfg[subconf]}
            cfg.pop(subconf)

    from src.utils.config import merge_cfg, make_paths_absolute
    cfg = make_paths_absolute(cfg)
    # load config
    cfg_orig, exp_folder = load_model_config_exp(checkpoint_path)
    # merge original experiment config with the current evaluation config
    cfg = merge_cfg(cfg, cfg_orig)
    cfg = merge_cfg(cfg, cfg['method_specs'])
    return cfg
