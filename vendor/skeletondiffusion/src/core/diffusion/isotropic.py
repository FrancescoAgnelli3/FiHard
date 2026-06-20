import torch
from torch.cuda.amp import autocast

from .base import LatentDiffusion, extract, default

class IsotropicGaussianDiffusion(LatentDiffusion):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - self.alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_cumprod - 1))
        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        alphas = 1. - self.betas

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod))
        
        # calculate loss weight
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)

        if self.objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif self.objective == 'pred_x0':
            loss_weight = snr
        elif self.objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    ########################################################################
    # FORWARD PROCESS
    #########################################################################  

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
        
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: self.get_white_noise(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
    ########################################################################
    # REVERSE PROCESS
    #########################################################################
        
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_combine_mean_var_noise(self, model_mean, model_log_variance, noise):
        return model_mean + (0.5 * model_log_variance).exp() * noise 
    
    def interpolate_noise(self, noise1, noise2, interpolate_funct=None, **kwargs):
        interpolated_noise = interpolate_funct(noise1, noise2)
        return interpolated_noise
        
    def p_interpolate_mean_var_noise(self, model_mean, model_log_variance, noise, noise2interpolate=None, **kwargs):
        interpolated_noise = self.interpolate_noise(noise, noise2interpolate,**kwargs) 
        return model_mean + (0.5 * model_log_variance).exp() * interpolated_noise 
    
