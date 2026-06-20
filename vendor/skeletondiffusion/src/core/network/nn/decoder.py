from typing import Tuple, Union

import torch
import torch.nn as nn

from ..layers import StaticGraphLinear, StaticGraphGRU, StaticGraphLSTM


class Decoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 feature_size: int, # 3
                 input_size: int, # bottleneck dim of h
                 hidden_size: int, # hidden size of the decoder
                 output_size: int,
                 node_types: torch.Tensor = None,
                 dec_num_layers: int = 1,
                 dropout: float = 0.,
                 param_groups=None,
                 recurrent_arch_decoder: str = 'StaticGraphGRU',
                 **kwargs):
        super().__init__()

        self.param_groups = param_groups
        self.num_layers = dec_num_layers
        self.if_consider_hip = kwargs["if_consider_hip"]
        self.activation_fn = torch.nn.Tanh()

    
        self.recurrent_arch = recurrent_arch_decoder
        recurrent_class = globals()[self.recurrent_arch]
        self.rnn = recurrent_class( feature_size + input_size,
                                  hidden_size,
                                  num_nodes=num_nodes,
                                  num_layers=dec_num_layers,
                                  learn_influence=True,
                                  node_types=node_types,
                                  recurrent_dropout=dropout,
                                  learn_additive_graph_influence=True,
                                  clockwork=False)
        
        self.initial_hidden_h = StaticGraphLinear(feature_size + input_size,
                                                  hidden_size,
                                                  num_nodes=num_nodes,
                                                  learn_influence=True,
                                                  node_types=node_types)
        if self.recurrent_arch == 'StaticGraphLSTM': 
            self.initial_hidden_c = StaticGraphLinear(feature_size + input_size,
                                                  hidden_size,
                                                  num_nodes=num_nodes,
                                                  learn_influence=True,
                                                  node_types=node_types)
        self.fc = StaticGraphLinear(hidden_size,
                                      output_size,
                                      num_nodes=num_nodes,
                                      learn_influence=True,
                                      node_types=node_types)
        
        self.dropout = nn.Dropout(dropout)
        
    def init_recurrent_hidden(self, x: torch.Tensor, h: torch.Tensor, z: torch.Tensor, state=None):
        x_t = x[:, -1]
        x_t_s = x[:, -1].clone()
        if state is None:
            x_t_1 = x[:, -2] # we are taking this one
        else:
            x_t_1 = state

        h_z = h

        # Initialize hidden state of rnn
        if self.recurrent_arch == 'StaticGraphGRU':
            rnn_h = self.initial_hidden_h(torch.cat([x_t_1, h_z], dim=-1))
            hidden = [(rnn_h, None)] * self.num_layers
        elif self.recurrent_arch == 'StaticGraphLSTM':
            rnn_h = self.initial_hidden_h(torch.cat([x_t_1, h_z], dim=-1))
            rnn_c = self.initial_hidden_c(torch.cat([x_t_1, h_z], dim=-1))
            hidden = [(rnn_h, rnn_c, None)] * self.num_layers
        else: 
            assert 0, f"architeture type {self.recurrent_arch} not supported"
        rec_input = torch.cat([x_t, h_z], dim=-1).unsqueeze(1)
        return rec_input, hidden 
        

    def forward(self, x: torch.Tensor, h: torch.Tensor, z: torch.Tensor,
                ph: int = 1, state=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = list()
        x_t_s = x[:, -1].clone()
        rec_input, hidden = self.init_recurrent_hidden(x=x, h=h, z=z, state=state)

        for i in range(ph):
            # Run RNN
            rnn_out, hidden = self.rnn(input=rec_input, states=hidden, t_i=i)  # [B * Z, 1, N, D]
            y_t = rnn_out.squeeze(1)  # [B * Z, N, D]
            y_t = self.dropout(y_t)

            y_t_state = self.fc(y_t)
            y_t_state= self.activation_fn(y_t_state)
            
            out.append(y_t_state)

        out = torch.stack(out, dim=1)

        return out, x_t_s
