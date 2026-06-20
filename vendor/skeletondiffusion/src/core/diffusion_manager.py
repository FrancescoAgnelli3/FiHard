from typing import Tuple, Optional, List, Union, Dict, Any
import torch

from .network import Denoiser
from .diffusion import IsotropicGaussianDiffusion, NonisotropicGaussianDiffusion, get_cov_from_corr


class DiffusionManager():
    def __init__(self, diffusion_type: str='IsotropicGaussianDiffusion', skeleton=None, covariance_matrix_type: str = 'adjacency', 
                 reachability_matrix_degree_factor=0.5, reachability_matrix_stop_at=0, if_sigma_n_scale=True, sigma_n_scale='spectral', if_run_as_isotropic=False, 
                 **kwargs):
        
        model = self.get_network(**kwargs)
        self.diffusion_type = diffusion_type

        if diffusion_type == 'NonisotropicGaussianDiffusion':
            # define SigmaN 
            if covariance_matrix_type == 'adjacency':
                correlation_matrix = skeleton.adj_matrix
            elif covariance_matrix_type == 'reachability':
                correlation_matrix = skeleton.reachability_matrix(factor=reachability_matrix_degree_factor, stop_at=reachability_matrix_stop_at)
            else: 
                assert 0, "Not implemented"
            N, *_ = correlation_matrix.shape

            Sigma_N, Lambda_N, U = get_cov_from_corr(correlation_matrix=correlation_matrix,  if_sigma_n_scale=if_sigma_n_scale, sigma_n_scale=sigma_n_scale, if_run_as_isotropic=if_run_as_isotropic, **kwargs)            
            self.diffusion = NonisotropicGaussianDiffusion(Sigma_N=Sigma_N, Lambda_N=Lambda_N, U=U, model=model, **kwargs)
        elif diffusion_type == 'IsotropicGaussianDiffusion':
            self.diffusion =  IsotropicGaussianDiffusion(model=model, **kwargs)
        else:
            assert 0, f"{diffusion_type} Not implemented"
        
    def get_diffusion(self):
        return self.diffusion
        
    def get_network(self, num_nodes, diffusion_conditioning=False, latent_size=96, node_types: torch.Tensor = None, diffusion_arch=Dict[str, Any], **kwargs): 

        if diffusion_conditioning:
            cond_dim = latent_size
        else: 
            cond_dim = 0
        
        model = Denoiser(dim=latent_size, cond_dim=cond_dim, out_dim=latent_size, channels=num_nodes, num_nodes=num_nodes, node_types=node_types,**diffusion_arch)

        return model