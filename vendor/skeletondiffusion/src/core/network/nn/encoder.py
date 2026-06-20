from typing import Tuple, Union

import torch
import torch.nn as nn
from typing import Union

from ..layers import StaticGraphLinear, StaticGraphGRU, GraphGRUState, StaticGraphLSTM, GraphLSTMState


class Encoder(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 node_types: torch.Tensor = None,
                 enc_num_layers: int = 1,
                 dropout: float = 0.,
                 encoder_act: str ="tanh",
                 recurrent_arch: str = 'StaticGraphGRU',
                 **kwargs):
        super().__init__()
        if encoder_act == "tanh":
            self.activation_fn = torch.nn.Tanh() #Tanh() #torch.nn.LeakyReLU()
        elif encoder_act == "identity":
            self.activation_fn = torch.nn.Identity()
        else: 
            assert 0, "not implemented"
        self.num_layers = enc_num_layers
        self.recurrent_arch = recurrent_arch
        recurrent_class = globals()[self.recurrent_arch]
        self.rnn = recurrent_class(input_size,
                                  hidden_size,
                                  num_layers=enc_num_layers,
                                  node_types=node_types,
                                  num_nodes=num_nodes,
                                  bias=True,
                                  clockwork=False,
                                  learn_influence=True)

        self.fc = StaticGraphLinear(hidden_size,
                                    output_size,
                                    num_nodes=num_nodes,
                                    node_types=node_types,
                                    bias=True,
                                    learn_influence=True)


        self.initial_hidden1 = StaticGraphLinear(input_size,
                                                 hidden_size,
                                                 num_nodes=num_nodes,
                                                 node_types=node_types,
                                                 bias=True,
                                                 learn_influence=True)
        if self.recurrent_arch == 'StaticGraphLSTM': 
            self.initial_hidden_c = StaticGraphLinear(input_size,
                                            hidden_size,
                                            num_nodes=num_nodes,
                                            node_types=node_types,
                                            bias=True,
                                            learn_influence=True)

        self.dropout = nn.Dropout(dropout)
    def init_recurrent_hidden(self, x: torch.Tensor, state=None) -> Union[GraphGRUState, GraphLSTMState]:
        if state is None:
            rnn_h = self.initial_hidden1(x[:, 0])
            #rnn_h2 = self.initial_hidden2(x[:, 0])
            if 'GRU' in self.recurrent_arch:
                state = [(rnn_h, None)] * self.num_layers #states: Optional[List[GraphGRUState]] = None, GraphGRUState=Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
            elif 'LSTM' in self.recurrent_arch:
                rnn_c = self.initial_hidden_c(x[:, 0])
                state = [(rnn_h, rnn_c, None)] * self.num_layers #states: Optional[List[GraphGRUState]] = None
                #Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            #Optional[List[GraphLSTMState]] = None
        return state
        
    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, Union[GraphGRUState, GraphLSTMState]]:
        # Initialize hidden state of rnn
        states = self.init_recurrent_hidden(x, state) # [B, T, N, 3] --> [B, N, D]
        y, state = self.rnn(input=x, states=states)  # [B, T, N, 3] --> [B, T, N, D]
        h = self.activation_fn(self.fc(self.dropout(y[:, -1])))  # [B, T, N, D] --> [B, N, D]
        return h, state
