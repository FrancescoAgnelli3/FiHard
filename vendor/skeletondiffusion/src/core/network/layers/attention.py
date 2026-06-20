import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum

from .graph_structural import StaticGraphLinear



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm((dim), elementwise_affine=True)

    def forward(self, x):
        x = torch.swapaxes(x, -2, -1)
        x = self.norm(x)
        x = torch.swapaxes(x, -2, -1)
        return x
    
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.g * (x.shape[-1] ** 0.5) #normalize divides by maximum norm element. Different from original in which we take the max norma nd not the sum of square elem. 

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
    
class Block(nn.Module):
    def __init__(self, dim, dim_out, norm_type='none', act_type='tanh', *args, **kwargs):
        super().__init__()
        self.proj = StaticGraphLinear(dim, dim_out, *args, **kwargs)
                                            # num_nodes=num_nodes,
                                            # node_types=T)
        if norm_type == 'none':
            self.norm = nn.Identity() #nn.GroupNorm(groups, dim_out)
        elif norm_type == 'layer':
            self.norm = LayerNorm(kwargs['num_nodes'])
        else: 
            assert 0, f"Norm type {norm_type} not implemented!"
        if act_type == 'tanh':
            self.act = nn.Tanh()
        else: 
            assert 0, f"Activation type {act_type} not implemented!"

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Tanh(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups = groups, **kwargs)
        self.block2 = Block(dim_out, dim_out, groups = groups, **kwargs)
        self.res_linear = StaticGraphLinear(dim, dim_out, bias=False, **kwargs)  if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b 1 c')
            scale_shift = time_emb.chunk(2, dim = -1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_linear(x)

# We need default num_heads: int = 8,
class Attention(nn.Module):
    def __init__(self, dim, dim_out=None, heads = 4, dim_head = 32,qkv_bias: bool = False, attn_dropout: float = 0., proj_dropout: float = 0., qk_norm: bool = False, norm_layer: nn.Module = nn.Identity, **kwargs):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        dim_out = dim_out if dim_out is not None else dim

        self.to_qkv = StaticGraphLinear(dim,hidden_dim * 3,bias=qkv_bias, **kwargs)
        self.to_out = StaticGraphLinear(hidden_dim,dim_out,bias=False,**kwargs)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(proj_dropout)


        self.q_norm = norm_layer(dim_head) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(dim_head) if qk_norm else nn.Identity()

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h c)  -> b h c n', h = self.heads), qkv)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        sim = einsum('b h c n, b h c j -> b h n j', q, k)
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h n j, b h d j -> b h n d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out_dropout(self.to_out(out))
    
    
