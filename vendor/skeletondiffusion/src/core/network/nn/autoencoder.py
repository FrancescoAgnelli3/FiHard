import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder

        
class AutoEncoder(nn.Module):
    def __init__(self,
                num_nodes: int,
                encoder_hidden_size: int,
                decoder_hidden_size: int,
                latent_size: int,
                node_types: torch.Tensor = None,
                input_size: int = 3,
                z_activation: str = 'tanh',
                enc_num_layers: int = 1,
                loss_pose_type: str = 'l1',
                **kwargs):
        super().__init__()
        self.param_groups = [{}]
        self.latent_size = latent_size
        self.loss_pose_type = loss_pose_type 

        self.encoder = Encoder(num_nodes=num_nodes,
                               input_size=input_size,
                               hidden_size=encoder_hidden_size,
                               output_size=latent_size,
                               node_types=node_types,
                               enc_num_layers = enc_num_layers,
                               recurrent_arch = kwargs['recurrent_arch_enc'],)

        
        assert kwargs['output_size'] == input_size
        self.decoder = Decoder( num_nodes=num_nodes,
                        input_size=latent_size ,
                        feature_size=input_size,
                        hidden_size=decoder_hidden_size,
                        node_types=node_types,
                        param_groups=self.param_groups,
                        **kwargs
                        )
        assert z_activation in ['tanh', 'identity'], f"z_activation must be either 'tanh' or 'identity', but got {z_activation}"
        self.z_activation = nn.Tanh() if z_activation == "tanh" else nn.Identity()

    
    def forward(self, x):
        h, _ = self.encoder(x)
        return h
    
    def get_past_embedding(self, past, state=None):
        with torch.no_grad():
            h_hat_embedding = self(past)
        z_past = self.z_activation(h_hat_embedding)
        return z_past
    
    def get_embedding(self, future, state=None):
        z = self.forward(future)
        return z
    
    def get_train_embeddings(self, y, past, state=None):
        z_past = self.get_past_embedding(past, state=state)
        z = self.get_embedding(y, state=state)
        return z_past, z
    
    def decode(self, x: torch.Tensor, h: torch.Tensor, z: torch.Tensor, ph=1, state=None):
        x_tiled = x[:, -2:] 
        out, _ = self.decoder(x=x_tiled,
                                h=h,
                                z=z,
                                ph=ph,
                                state=state)  # [B * Z, T, N, D]
        return out

    def autoencode(self, y, past, ph=1, state=None):
        z_past, z = self.get_train_embeddings(y, past, state=state)
        out = self.decode(past, z, z_past, ph)
        return out, z_past, z

    def loss(self, y_pred, y, type=None, reduction="mean", **kwargs):
        type = self.loss_pose_type if type is None else type
        if type=="mse":
            out = torch.nn.MSELoss(reduction="none")(y_pred,y)
        elif type in ["l1", "L1"]:
            out = torch.nn.L1Loss(reduction="none")(y_pred,y)
        else: 
            assert 0, "Not implemnted"
        loss = (out.sum(-1) #spatial size 
                    .mean(-1) #keypoints
                    .mean(-1) # timesteps
                    )
        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        else: 
            assert 0, "Not implemnted"
        return loss

    
    

