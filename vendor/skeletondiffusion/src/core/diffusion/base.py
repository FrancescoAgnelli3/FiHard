import math
from random import random
from functools import partial
from collections import namedtuple
from typing import Tuple, Optional, List, Union, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import reduce


from tqdm.auto import tqdm


# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exp_beta_schedule(timesteps, factor=3.0):
    steps = timesteps + 1
    x = torch.linspace(-factor, 0, steps, dtype = torch.float64)#/timesteps
    betas = torch.exp(x)
    return torch.clip(betas, 0, 0.999)


class LatentDiffusion(nn.Module):
    def __init__(self,
        model:torch.nn.Module, latent_size=96, diffusion_timesteps=10, diffusion_objective="pred_x0", sampling_timesteps=None, diffusion_activation='identity', 
        diffusion_conditioning=False, diffusion_loss_type='mse',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        beta_schedule_factor=3.0,
        ddim_sampling_eta = 0.,
        **kwargs
    ):
        
        super().__init__()

        
        if diffusion_activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif diffusion_activation == "identity":
            self.activation = torch.nn.Identity()
        self.silent = True
        self.condition = diffusion_conditioning
        self.loss_type = diffusion_loss_type
        
        self.statistics_pred = None
        self.statistics_obs = None

        
        timesteps=diffusion_timesteps
        objective=diffusion_objective
        
        
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = latent_size
        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'exp':
            betas = exp_beta_schedule(timesteps, beta_schedule_factor)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))       

        print("Diffusion is_ddim_sampling: ", self.is_ddim_sampling)

        
   
    def set_normalization_statistics(self, statistics_pred, statistics_obs):
        self.statistics_pred = statistics_pred
        self.statistics_obs = statistics_obs
        print("Setting normalization statistics for diffusion")    
        
    def get_white_noise(self, x, *args, **kwargs):
        return self.get_noise(x, *args, **kwargs)
    
    def get_start_noise(self, x, *args, **kwargs):
        return self.get_white_noise(x, *args, **kwargs)
    
    def get_noise(self, x, *args, **kwargs):
        """
        x is either tensor or shape 
        """
        if torch.is_tensor(x):
            return torch.randn_like(x)
        elif isinstance(x, tuple):
            return torch.randn(*x, *args, **kwargs) 
        
    #######################################################################
    # TO SUBCLASS
    #######################################################################

    def predict_start_from_noise(self, x_t, t, noise):
        assert 0, "Not implemented"
        ...
        return x_t

    def predict_noise_from_start(self, x_t, t, x0):
        assert 0, "Not implemented"
        ...
        return x_t

    def predict_v(self, x_start, t, noise):
        assert 0, "Not implemented"
        ...
        return x_start

    def predict_start_from_v(self, x_t, t, v):
        assert 0, "Not implemented"
        ...
        return x_t

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        assert 0, "Not implemented"
        ...
        return x_start

    def q_posterior(self, x_start, x_t, t):
        assert 0, "Not implemented"
        ...
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_combine_mean_var_noise(self, model_mean, model_log_variance, noise):
        assert 0, "Not implemented"
        ...
        return model_mean 
    
    def p_interpolate_mean_var_noise(self, model_mean, model_log_variance, noise, node_idx:Optional[int] = None, interpolate_factor=0.0, noise2interpolate=None, **kwargs):
        assert 0, "Not implemented"
        ...
        return model_mean 
    
    def loss_funct(self, model_out, target, *args, **kwargs):
        if self.loss_type == "mse":
            loss = F.mse_loss(model_out, target, reduction = 'none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(model_out, target, reduction = 'none')
        else: 
            assert 0, "Not implemented"
        return loss
    
    ########################################################################
    # NETWORK INTERFACE
    ######################################################################### 


    def model_predictions(self, x, t, x_self_cond = None, x_cond=None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.feed_model(x, t, x_self_cond=x_self_cond, x_cond=x_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start) 
    
    def feed_model(self, x, t, x_self_cond = None, x_cond=None): 
        if self.condition:
            assert x_cond is not None
            if x.shape[0] > x_cond.shape[0]:
                # training with multiple samples
                x_cond = x_cond.repeat_interleave( int(x.shape[0]/x_cond.shape[0]), 0)
            model_in = x 
        else: 
            model_in = x

        model_output = self.model(model_in, t, x_self_cond, x_cond)
        model_output = self.activation(model_output)
        return model_output
    
    ########################################################################
    # FORWARD PROCESS
    #########################################################################  
    
    
    def p_losses(self, x_start, t, noise = None, x_cond=None, n_train_samples=1):
        b, c, n = x_start.shape
        if n_train_samples > 1:
            x_start = x_start.repeat_interleave(n_train_samples, dim=0)
            t = t.repeat_interleave(n_train_samples, dim=0)
            if x_cond is not None:
                x_cond = x_cond.repeat_interleave(n_train_samples, dim=0)
        noise = default(noise, self.get_white_noise(x_start, t)) # noise for timesteps t

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, x_cond=x_cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.feed_model(x, t, x_self_cond=x_self_cond, x_cond=x_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_funct(model_out, target, t)
        loss = reduce(loss, 'b ... -> b', 'mean') # [batch*n_train_samples, Nodes, latent_dim] -> [batch*n_train_samples]

        return loss,  extract(self.loss_weight, t.view(b, -1)[:, 0], loss.shape[0:1]), model_out
    
    def forward(self, x, *args, x_cond=None, **kwargs):
        b, c, n, device, seq_length, = *x.shape, x.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, *args, x_cond=x_cond, **kwargs)
    
    
    ########################################################################
    # REVERSE PROCESS
    #########################################################################
    
    def p_mean_variance(self, x, t, x_self_cond = None, x_cond=None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond, x_cond=x_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True, sampling_noise=None, *args, if_interpolate=False, noise2interpolate=None, interpolation_kwargs:Dict=None, **kwargs):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, *args, **kwargs)
        
        if sampling_noise is not None and t > 0:
            noise = sampling_noise[:, sampling_noise.shape[1]-t]
        else:
            noise = self.get_white_noise(x) if t > 0 else 0. # no noise if t == 0
        
        if if_interpolate and t > 0:
            noise2 = noise2interpolate[:, sampling_noise.shape[1]-t]
            assert noise2.shape == noise.shape
            pred_img = self.p_interpolate_mean_var_noise(model_mean, model_log_variance, noise, noise2, **interpolation_kwargs)
        else:
            pred_img = self.p_combine_mean_var_noise(model_mean, model_log_variance, noise)
        return pred_img, x_start, noise, model_mean

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond=None, start_noise=None, sampling_noise=None, return_sampling_noise=False, return_timages=False, **kwargs):
        batch, device = shape[0], self.betas.device
        if start_noise is not None:
            assert start_noise.shape == shape, f"Shape mismatch: {start_noise.shape} != {shape}"
            img = start_noise
            noise = start_noise.clone()
        else:
            img = self.get_start_noise(shape, device=device)
            noise = img.clone()
            
        if sampling_noise is not None:
            assert sampling_noise.shape[2:] == shape[1:], f"Shape mismatch: {start_noise.shape} != {shape}"
            assert sampling_noise.shape[0] == shape[0], f"Shape mismatch: {start_noise.shape} != {shape}"
            assert sampling_noise.shape[1] == self.num_timesteps - 1

        x_start = None
        imgs = []
        noise_t = []
        mean_t = []
        if not self.silent:
            print(f"Evaluation with {len(range(0, self.num_timesteps))} diffusion steps")
        for t in reversed(range(0, self.num_timesteps)): #, desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start, nt, model_mean = self.p_sample(img, t, self_cond, x_cond=x_cond, sampling_noise=sampling_noise, **kwargs)
            if return_sampling_noise and t!=0:
                noise_t.append(nt)
                mean_t.append(model_mean)
            if return_timages and t!=0:
                imgs.append(img)


        if return_sampling_noise:
            noise_t = torch.stack(noise_t, dim=1)
            mean_t = torch.stack(mean_t, dim=1)
        if return_timages:
            print("Returning timages")
            imgs = torch.stack(imgs, dim=1)

        if return_sampling_noise:
            if return_timages:
                noise = (noise, noise_t, imgs)
            else:
                noise = (noise, noise_t, mean_t)
        else:
            if return_timages:
                noise = (noise, imgs)
        return img, noise
   
    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True, x_cond=None, start_noise=None): 
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        if start_noise is not None:
            assert start_noise.shape == shape
            img = start_noise
            noise = start_noise.clone()
        else:
            img = torch.randn(shape, device=device)
            noise = img.clone()

        imgs = []

        x_start = None
        if not self.silent:
            print(f"Evaluation with {len(time_pairs)} diffusion steps")
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, x_cond=x_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                # imgs.append(img)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sqrt_alpha_next = self.sqrt_alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt() #eta*beta_t_tilde
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * sqrt_alpha_next + \
                  c * pred_noise + \
                  sigma * noise
            # imgs.append(img)

        # imgs = torch.stack(imgs, dim=1)
        return img, noise
    
    
    @torch.no_grad()
    def sample(self, batch_size = 16, *args, **kwargs):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length),*args, **kwargs)


