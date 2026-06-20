import torch
import os
import yaml

from .core import DiffusionManager
from .inference_utils import create_model, load_model_config_exp
from .utils.load import load_model_checkpoint


def load_model_config_exp_autoenc(checkpoint_path):
    exp_folder = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(exp_folder, 'config.yaml')
    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg, exp_folder
    

def get_eval_out_folder(exp_folder, checkpoint_path, data_split, cfg):
    checkpoint = checkpoint_path.split("/")[-1].split('_val')[0].split("_")[-1].replace(".pt", "")
    long_term_label = '' if not cfg['if_long_term_test']  and cfg['long_term_factor'] > 1 else '_longterm'+str(cfg['long_term_factor'])
    noise_label = f"_noisyobs{cfg['noise_level']}-{cfg['noise_std']}" if cfg['if_noisy_obs'] else ""
    stats_folder = os.path.join(exp_folder, f"eval_{cfg['dataset_name']}_{cfg['batch_size']}{long_term_label}{noise_label}", data_split,  f"{torch.cuda.get_device_name(0).replace(' ', '_')}_seed{cfg['seed']}", f"checkpoint{checkpoint}")
    os.makedirs(stats_folder, exist_ok=True)
    return stats_folder

def prepare_autoencoder(config, skeleton, silent=False):
    for i in range(torch.cuda.device_count()):
        if not silent:
            print(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    
    # build model architecture
    config['device'] = device
    model = create_model(node_types=skeleton.nodes_type_id, num_nodes=skeleton.num_nodes, **config, **config['autoenc_arch'])

    # Load models
    checkpoint_path = config['pretrained_autoencoder_path'] if 'pretrained_autoencoder_path' in config else config['checkpoint_path']
    if not silent:
        print('Loading Autoencoder checkpoint: {} ...'.format(checkpoint_path))

    checkpoint_model = load_model_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint_model['model'])

    if 'n_gpu' in config and config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    
    return model, device

def prepare_model(config, skeleton, silent=False,  **kwargs):
    """
    This function must take as input a config_file and at least a second arg (skeleton) and return a model and a device (cpu or cuda)
    model, device = prepare_model(config, skeleton,  **kwargs)
    """

    model, device = prepare_autoencoder(config, skeleton, silent=silent)

    # build model architecture
    diffusionmanager = DiffusionManager(skeleton=skeleton, num_nodes=skeleton.num_nodes, node_types=skeleton.nodes_type_id, **kwargs)
    diffusion = diffusionmanager.get_diffusion().to(device)
    # Load models
    if not silent:
        print('Loading Diffusion checkpoint: {} ...'.format(config['checkpoint_path']))

    checkpoint_diffusion = load_model_checkpoint(config['checkpoint_path'])
    # if  config['checkpoint_path'].endswith('.pth.tar'):
    #     diffusion.load_state_dict(checkpoint_diffusion)
    diffusion.load_state_dict(checkpoint_diffusion['model'])

    # in case you renamed nn modules and want to update checkpoint
    # diffusion.load_state_dict(checkpoint_diffusion['model'], strict=False)
    # checkpoint_diffusion['model'] = diffusion.state_dict() 
    # torch.save(checkpoint_diffusion, os.path.join(os.path.dirname(config['checkpoint_path']), "final_ckpt.pt"))

    if 'n_gpu' in config and config['n_gpu'] > 1:
        diffusion = torch.nn.DataParallel(diffusion)

    diffusion = diffusion.to(device)
    diffusion.eval()
    
    return (model, diffusion), device, diffusionmanager 



def get_diffusion_latent_codes(obs, model, num_samples=50, **kwargs):
    model, diffusion = model
    bs, obs_length, j, f = obs.shape
    sampler_kwargs = {} if 'sampler_kwargs' not in kwargs else kwargs['sampler_kwargs']

    past_embedding = model.get_past_embedding(obs)
    if kwargs['diffusion_conditioning']:
        z_past = past_embedding.repeat_interleave(num_samples, 0)
        latent_pred, _ = diffusion.sample(batch_size=bs*num_samples, x_cond=z_past, **sampler_kwargs)
        z_past = past_embedding

    else: 
        latent_pred, _ = diffusion.sample(batch_size=bs*num_samples, **sampler_kwargs) # out_steps has shape -> [N, Joints, Feat]
        z_past = model.z_activation(z_past)

    return latent_pred, z_past

def decode_latent_pred(obs, latent_pred, z_past, model, num_samples=50,  pred_length=100, **kwargs):
    model, diffusion = model
    bs, obs_length, j, f = obs.shape
    obs = obs.repeat_interleave(num_samples, 0)
    z_past = z_past.repeat_interleave(num_samples, 0)
    z_past = z_past.view(bs*num_samples, *z_past.shape[1:])
    pred = model.decode(obs, latent_pred, z_past, ph=pred_length) # [N, t, Joints, 3]

    pred =pred.view(bs, num_samples, pred_length, j, f)
    
    return pred
    
def get_prediction(obs, model, num_samples=50, pred_length=100, **kwargs):
    lat_pred, z_past = get_diffusion_latent_codes(obs, model, num_samples=num_samples, **kwargs)
    pred = decode_latent_pred(obs, lat_pred, z_past, model, num_samples=num_samples, pred_length=pred_length, **kwargs)
    return pred


def process_evaluation_pair(skeleton, target, pred_dict):
    pred, mm_gt, obs = pred_dict['pred'], pred_dict['mm_gt'] if 'mm_gt' in pred_dict else None, pred_dict['obs']
    target = skeleton.transform_to_metric_space(target)
    pred = skeleton.transform_to_metric_space(pred)
    obs = skeleton.transform_to_metric_space(obs)
    mm_gt = [skeleton.transform_to_metric_space(gt) for gt in mm_gt] if mm_gt is not None else None
    
    batch_size, n_samples, seq_length, num_joints, features = pred.shape
    # batch_size, n_samples, seq_length, num_joints, features = pred.shape
    assert features == 3 and list(target.shape) == [batch_size, seq_length, num_joints, features]
    return target, pred, mm_gt, obs
   
 