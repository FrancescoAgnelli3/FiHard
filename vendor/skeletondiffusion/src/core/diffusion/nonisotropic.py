import torch
from torch.cuda.amp import autocast
from .base import LatentDiffusion, extract, default

def extract_matrix(matrix, t, x_shape):
    b, *_ = t.shape
    T, N, *_ = matrix.shape
    out = torch.index_select(matrix, 0, t)
    out = out.reshape(b, *out.shape[1:])
    while len(x_shape) > len(out.shape):
        out = out.unsqueeze(-1)
    return out

def matmul_matrix_diagmatrix(matrix, diagonal_vector):
    """
    matrix: (B, T, N, N)
    diagonal_vector: (B, T, N)
    """
    return matrix * diagonal_vector.unsqueeze(-2)

def matmul_diagmatrix_matrix(diagonal_vector, matrix):
    """
    matrix: (B, T, N, N)
    diagonal_vector: (B, T, N)
    """
    return diagonal_vector.unsqueeze(-1) * matrix

def verify_noise_scale(diffusion):
    N, *_ = diffusion.Lambda_N.shape
    alphas = 1 - diffusion.betas
    noise = diffusion.get_noise((2000, diffusion.num_timesteps, N))
    zeta_noise = torch.sqrt(diffusion.Lambda_t.unsqueeze(0)) * noise
    print("current: ", (zeta_noise**2).sum(-1).mean(0))
    print("original standard gaussian diffusion: ",(1-alphas) * zeta_noise.shape[-1])

def compute_covariance_matrices(diffusion: torch.nn.Module,  Lambda_N: torch.Tensor,  diffusion_covariance_type='skeleton-diffusion', gamma_scheduler = 'cosine'):
    N, *_  = Lambda_N.shape
    alphas = 1. - diffusion.betas
    def _alpha_sumprod(alphas, t):
        return torch.sum(torch.cumprod(torch.flip(alphas[:t+1], [0]), dim=0))
    alphas_sumprod = torch.stack([_alpha_sumprod(alphas, t) for t in range(len(alphas))], dim=0)
    diffusion.alphas_sumprod = alphas_sumprod
    if diffusion_covariance_type == 'isotropic':
        assert (Lambda_N == 0).all()
        Lambda_t = (1-alphas).unsqueeze(-1) # (Tdiff, N)
        Lambda_bar_t = (1-diffusion.alphas_cumprod.unsqueeze(-1))
        Lambda_bar_t_prev = torch.cat([torch.zeros(1).unsqueeze(0), Lambda_bar_t[:-1]], dim=0)
    elif diffusion_covariance_type == 'anisotropic': 
        Lambda_t = (1-alphas.unsqueeze(-1))*Lambda_N # (Tdiff, N)
        Lambda_bar_t = (1-diffusion.alphas_cumprod.unsqueeze(-1))*Lambda_N
        Lambda_bar_t_prev = (1-diffusion.alphas_cumprod_prev.unsqueeze(-1))*Lambda_N
    elif diffusion_covariance_type == 'skeleton-diffusion':
        if gamma_scheduler== 'cosine':
            gammas = 1 - alphas
        elif gamma_scheduler == 'mono_decrease':
            gammas = 1 - torch.arange(0, diffusion.num_timesteps)/diffusion.num_timesteps
        else: 
            assert 0, "Not implemented"
        Lambda_I = Lambda_N - 1
        gammas_bar = (1-alphas)*gammas
        gammas_tilde = diffusion.alphas_cumprod*torch.cumsum(gammas_bar/diffusion.alphas_cumprod, dim=-1)
        Lambda_t = Lambda_I.unsqueeze(0)*gammas_bar.unsqueeze(-1) + (1-alphas).unsqueeze(-1) # (Tdiff, N)
        Lambda_bar_t = Lambda_I.unsqueeze(0)*gammas_tilde.unsqueeze(-1)  + (1-diffusion.alphas_cumprod.unsqueeze(-1))
        Lambda_bar_t_prev = torch.cat([torch.zeros(N).unsqueeze(0), Lambda_bar_t[:-1]], dim=0) # we start from det so it must be zero for t=-1
    else:
        assert 0, "Not implemented"
        
    return Lambda_t, Lambda_bar_t, Lambda_bar_t_prev


class NonisotropicGaussianDiffusion(LatentDiffusion):
    def __init__(self, Sigma_N: torch.Tensor, Lambda_N: torch.Tensor, U: torch.Tensor, diffusion_covariance_type='skeleton-diffusion', loss_reduction_type='l1', gamma_scheduler = 'cosine',  **kwargs):
        super().__init__( **kwargs)
        alphas = 1. - self.betas


        N, _ = Sigma_N.shape
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('Lambda_N', Lambda_N)
        register_buffer('Sigma_N', Sigma_N)
        self.set_rotation_matrix(U)


        Lambda_t, Lambda_bar_t, Lambda_bar_t_prev = compute_covariance_matrices(diffusion=self, Lambda_N=Lambda_N,  
                                                                                diffusion_covariance_type=diffusion_covariance_type, gamma_scheduler=gamma_scheduler)

        def create_diagonal_matrix(diagonal_vector):
          return torch.stack([torch.diag(diag)  for diag in diagonal_vector], dim=0) # [T, N, N]
        ######### forward , for training and inference #####################
        #predict_noise_from_start
        inv_sqrt_Lambda_bar = 1/torch.sqrt(Lambda_bar_t)
        inv_sqrt_Lambda_bar_sqrt_alphas_cumprod = (1/torch.sqrt(Lambda_bar_t))*self.sqrt_alphas_cumprod.unsqueeze(-1)
        register_buffer('inv_sqrt_Lambda_bar_mmUt', matmul_diagmatrix_matrix(inv_sqrt_Lambda_bar, self.U_transposed.unsqueeze(0))) #create_diagonal_matrix(inv_sqrt_Lambda_bar)@self.U_transposed.unsqueeze(0))
        register_buffer('inv_sqrt_Lambda_bar_sqrt_alphas_cumprod_mmUt', matmul_diagmatrix_matrix(inv_sqrt_Lambda_bar_sqrt_alphas_cumprod, self.U_transposed.unsqueeze(0))) # create_diagonal_matrix(inv_sqrt_Lambda_bar_sqrt_alphas_cumprod)@self.U_transposed.unsqueeze(0))
        #predict_start_from_noise
        sqrt_Lambda_bar = torch.sqrt(Lambda_bar_t)
        sqrt_Lambda_bar_sqrt_recip_alphas_cumprod = torch.sqrt(Lambda_bar_t/self.alphas_cumprod.unsqueeze(-1))
        register_buffer('Umm_sqrt_Lambda_bar_t', matmul_matrix_diagmatrix(U.unsqueeze(0), sqrt_Lambda_bar))
        register_buffer('Umm_sqrt_Lambda_bar_t_sqrt_recip_alphas_cumprod', matmul_matrix_diagmatrix(U.unsqueeze(0), sqrt_Lambda_bar_sqrt_recip_alphas_cumprod))

        ######### q_posterior , for reverse process #####################
        #q_posterior
        Lambda_posterior_t = Lambda_t*Lambda_bar_t_prev*(1/Lambda_bar_t)
        sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        register_buffer('Lambda_posterior', Lambda_posterior_t)
        register_buffer('Lambda_posterior_log_variance_clipped', torch.log(Lambda_posterior_t.clamp(min =1e-20)))
        
        posterior_mean_coef1_x0 = sqrt_alphas_cumprod_prev.unsqueeze(-1).unsqueeze(-1)*(U.unsqueeze(0)@create_diagonal_matrix((1/Lambda_bar_t)*Lambda_t)@self.U_transposed.unsqueeze(0))
        posterior_mean_coef2_xt = torch.sqrt(alphas).unsqueeze(-1).unsqueeze(-1)*(U.unsqueeze(0)@create_diagonal_matrix((1/Lambda_bar_t)*Lambda_bar_t_prev)@self.U_transposed.unsqueeze(0))
        register_buffer('posterior_mean_coef1_x0', posterior_mean_coef1_x0)
        register_buffer('posterior_mean_coef2_xt', posterior_mean_coef2_xt)

        ######### loss #####################
        self.loss_reduction_type = loss_reduction_type         
        sqrt_recip_Lambda_bar_t = torch.sqrt(1. / Lambda_bar_t)           
        register_buffer('mahalanobis_S_sqrt_recip', matmul_diagmatrix_matrix(sqrt_recip_Lambda_bar_t, self.U_transposed.unsqueeze(0))) # create_diagonal_matrix(sqrt_recip_Lambda_bar_t)@self.U_transposed.unsqueeze(0))

        if self.objective == 'pred_noise':
            loss_weight = torch.ones_like(alphas)
        elif self.objective == 'pred_x0':
            loss_weight = self.alphas_cumprod
        elif self.objective == 'pred_v':
            assert 0, "Not implemented"
            # loss_weight = snr / (snr + 1)
        register_buffer('loss_weight', loss_weight)
    
        assert not  len(self.mahalanobis_S_sqrt_recip.shape) == 1
            

    ########################################################################
    # CLASS FUNCTIONS
    #########################################################################  
    
    def set_rotation_matrix(self, U:torch.Tensor):
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('U', U)    
        register_buffer('U_transposed', U.t())    
        
    def check_eigh(self):
        return torch.isclose(self.U@torch.diag(self.Lambda_N)@self.U_transposed,self.Sigma_N)#.all(), "U@Lambda_N@U^T must be equal to Sigma_N"

    def get_anisotropic_noise(self, x, *args, **kwargs):
        """
        x is either tensor or shape 
        """
        return self.get_noise(x, *args, **kwargs)*self.Lambda_N.unsqueeze(-1)
    
    ########################################################################
    # FORWARD PROCESS
    #########################################################################        

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: self.get_white_noise(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_matrix(self.Umm_sqrt_Lambda_bar_t, t, x_start.shape) @ noise
        )
    # for inference    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_matrix(self.Umm_sqrt_Lambda_bar_t_sqrt_recip_alphas_cumprod, t, x_t.shape) @ noise
        )
    # for inference    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract_matrix(self.inv_sqrt_Lambda_bar_mmUt, t, x_t.shape)@x_t -\
                extract_matrix(self.inv_sqrt_Lambda_bar_sqrt_alphas_cumprod_mmUt, t, x_t.shape)@x0
        )
        
    ########################################################################
    # LOSS
    ######################################################################### 
       
    def mahalanobis_dist(self, matrix, vector):
        return (matrix@vector).abs() # check shape
    
    def loss_funct(self, model_out, target, t):
        difference = target - model_out if self.objective == 'pred_noise' else model_out - target

        loss = self.mahalanobis_dist(extract_matrix(self.mahalanobis_S_sqrt_recip, t, difference.shape), difference)
        if self.loss_reduction_type == 'l1': 
            loss = loss
        elif self.loss_reduction_type == 'mse': 
            loss = loss**2
        else:
            assert 0, "Not implemented"
        return loss

    ########################################################################
    # REVERSE PROCESS
    #########################################################################
        
    def q_posterior_mean(self, x_start, x_t, t):
         return (
                extract_matrix(self.posterior_mean_coef1_x0, t, x_t.shape) @ x_start +
                extract_matrix(self.posterior_mean_coef2_xt, t, x_t.shape) @ x_t
            )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.q_posterior_mean(x_start, x_t, t)
        posterior_variance = extract_matrix(self.Lambda_posterior, t, x_t.shape)
        posterior_log_variance_clipped = extract_matrix(self.Lambda_posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_combine_mean_var_noise(self, model_mean, posterior_log_variance, noise):
        """ mean is in not diagonal coordinate system, posterior_log_variance is in diagonal coordinate system"""
        return model_mean + self.U@((0.5 * posterior_log_variance).exp() * noise)

    

    ########################################################################
    # INTERPOLATION
    #########################################################################

    def interpolate_noise(self, noise1, noise2, posterior_log_variance=None, interpolate_funct=None):
        noise1 = self.U@((0.5 * posterior_log_variance).exp() * noise1)
        noise2 = self.U@((0.5 * posterior_log_variance).exp() * noise2)
        interpolated_noise = interpolate_funct(noise1, noise2)
        return interpolated_noise
        
    
    def p_interpolate_mean_var_noise(self, model_mean, model_log_variance, noise, noise2interpolate=None, **kwargs):
        interpolated_noise = self.interpolate_noise(noise, noise2interpolate, posterior_log_variance=model_log_variance, **kwargs) 
        return model_mean + interpolated_noise # (0.5 * model_log_variance).exp()

