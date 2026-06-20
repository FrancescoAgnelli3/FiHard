
import yaml
import os
from datetime import datetime
import socket

def get_timestamp_filename():
    now = datetime.now()
    month = now.strftime("%B")
    dt_string = now.strftime("%d_%H-%M-%S")
    name = month[:3] + dt_string + f"_{socket.gethostname().split('.')[0]}"
    return name

def load_and_merge_autoenc_cfg(cfg, eval=False):
    config_gm = os.path.join(os.path.dirname(os.path.dirname(cfg['pretrained_autoencoder_path'])), 'config.yaml')
    with open(config_gm, 'r') as stream:
        cfg_gm = yaml.safe_load(stream)
    if eval:
        cfg_gm.pop('dataset', None)
    cfg = merge_cfg(cfg, cfg_gm)
    return cfg

def merge_cfg(cfg, cfg_gm):
    """
    If keys ar ein both cfg, we take the ones of cfg: cfg overrides cfg_gm
    """
    for k in set(cfg.keys()).intersection(cfg_gm.keys()):
        cfg_gm.pop(k, None)
    assert len(set(cfg.keys()).intersection(cfg_gm.keys())) == 0
    cfg = {**cfg, **cfg_gm}
    return cfg

def make_paths_absolute(cfg):
    cfg = cfg.copy()
    # transform paths in cfg to absolute paths
    for k in cfg:
        if 'data_loader' in k:
            if 'annotations_folder' in cfg[k]:
                cfg[k]['annotations_folder'] = os.path.abspath(cfg[k]['annotations_folder'])
            if 'segments_path' in cfg[k]:
                cfg[k]['segments_path'] = os.path.abspath(cfg[k]['segments_path'])

    if 'precomputed_folder' in cfg:
        cfg['precomputed_folder'] = os.path.abspath(cfg['precomputed_folder'])
    if 'annotations_folder' in cfg:
        cfg['annotations_folder'] = os.path.abspath(cfg['annotations_folder'])
    if 'pretrained_autoencoder_path' in cfg:
        cfg['pretrained_autoencoder_path'] = os.path.abspath(cfg['pretrained_autoencoder_path'])
    return cfg

def init_obj(config, name, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = config.init_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = config[name]
    module_args = config #dict(config[name]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args = {**kwargs, **config}
    return getattr(module, module_name)(*args, **module_args)

def flat_hydra_config(cfg):
    """
    Flatten the main dict categories of the Hydra config object into a single one.
    """
    for subconf in ['model', 'task', 'dataset', 'cov_matrix']:
        if subconf in cfg:
            cfg = {**cfg, **cfg[subconf]}
            cfg.pop(subconf)
    return cfg


def config_exp_folder(cfg):
    cfg['load_path'] = ""
    if not 'output_log_path' in cfg:
        dt_str = get_timestamp_filename() #datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not cfg['info'] == '':
            info_str = f"_{cfg['info']}"
        else:
            info_str = ''
        output_path = os.path.join(os.path.dirname(os.path.dirname(cfg['pretrained_autoencoder_path'])))
        cfg['output_log_path'] = os.path.join(output_path, f"{dt_str}{info_str}")
        os.makedirs(os.path.join(cfg['output_log_path'], 'checkpoints'), exist_ok=True)
    # Copy code and hydra configuration to experiment folder
    import shutil
    from omegaconf import OmegaConf
    code_dest_folder = os.path.join(cfg['output_log_path'], 'code')
    if not os.path.exists(code_dest_folder):
        os.makedirs(cfg['output_log_path'], exist_ok=True)
        with open(os.path.join(cfg['output_log_path'], 'config.yaml'), 'w') as config_file:
            OmegaConf.save(cfg, config_file)
        shutil.copytree(os.path.dirname(__file__), code_dest_folder)
    return cfg