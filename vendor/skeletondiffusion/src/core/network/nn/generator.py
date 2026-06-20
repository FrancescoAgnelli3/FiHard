import torch
from torch import nn
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb

from ..layers import StaticGraphLinear, Attention, ResnetBlock, Residual, PreNorm


class Denoiser(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        channels: int,
        cond_dim: int = 0,
        depth=1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        use_attention = True,
        **kwargs
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        diffusion_size = dim + cond_dim
        input_dim = dim* (2 if self_condition else 1) + cond_dim
        self.init_lin = StaticGraphLinear(input_dim, diffusion_size, bias=True, **kwargs)


        # time embeddings
        time_dim = (dim + cond_dim) * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(diffusion_size, theta = sinusoidal_pos_emb_theta)
            fourier_dim = diffusion_size

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.layers = nn.ModuleList([])
        for i in range(depth):

            self.layers.append(nn.ModuleList([
                                ResnetBlock(diffusion_size, diffusion_size,
                                                time_emb_dim = time_dim, groups = resnet_block_groups, # bias=True,
                                                **kwargs),
                                Residual(PreNorm(diffusion_size, Attention(diffusion_size, heads = attn_heads, dim_head = attn_dim_head,**kwargs))) if use_attention else Residual(PreNorm(diffusion_size, StaticGraphLinear(diffusion_size, diffusion_size,bias=False, **kwargs))),
                                ]))            
            layerlist = [ResnetBlock(diffusion_size, diffusion_size,
                                                time_emb_dim = time_dim, groups = resnet_block_groups, # bias=True,
                                                **kwargs),
                                ]
            if i!=depth-1:
                layerlist.append(
                                Residual(PreNorm(diffusion_size, Attention(diffusion_size, heads = attn_heads, dim_head = attn_dim_head,**kwargs))) if use_attention else Residual(PreNorm(diffusion_size, StaticGraphLinear(diffusion_size, diffusion_size,bias=False, **kwargs))),
                )
            else: 
                layerlist.append(nn.Identity())
            self.layers.append(nn.ModuleList(layerlist))


        out_dim = out_dim * (1 if not learned_variance else 2)
        self.final_res_block = ResnetBlock(diffusion_size*2, diffusion_size,
                                                time_emb_dim = time_dim, groups = resnet_block_groups, # bias=True,
                                                **kwargs)
        self.final_glin = StaticGraphLinear(diffusion_size, out_dim, bias=True, **kwargs) 
        
    def forward(self, x, time, x_self_cond = None, x_cond=None):

        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim = -1)
        if x_cond is not None:
            x = torch.cat([x_cond, x], dim=-1)

        x = self.init_lin(x)
        r = x.clone()

        t = self.time_mlp(time)


        for block1, attn in self.layers:
            x = block1(x, t)
            x = attn(x)

        x = torch.cat((x, r), dim = -1)

        x = self.final_res_block(x, t)
        return self.final_glin(x)