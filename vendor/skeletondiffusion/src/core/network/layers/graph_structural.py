from typing import Tuple, Optional, List, Union

import torch
from torch.nn import *
import math

def gmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ndo,bnd->bno', w, x)


class GraphLinear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)
        #if self.learn_influence:
        #    self.G.data.uniform_(-stdv, stdv)
        if len(self.weight.shape) == 3:
            self.weight.data[1:] = self.weight.data[0]
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        if g is None and self.learn_influence:
            g = torch.nn.functional.normalize(self.G, p=1., dim=1)
            #g = torch.softmax(self.G, dim=1)
        elif g is None:
            g = self.G
        w = self.weight[self.node_type_index]
        output = self.mm(input, w.transpose(-2, -1))
        if self.bias is not None:
            bias = self.bias[self.node_type_index]
            output += bias
        output = g.matmul(output)

        return output


class DynamicGraphLinear(GraphLinear):
    def __init__(self, num_node_types: int = 1, *args):
        super().__init__(*args)

    def forward(self, input: torch.Tensor, g: torch.Tensor = None, t: torch.Tensor = None) -> torch.Tensor:
        assert g is not None or t is not None, "Either Graph Influence Matrix or Node Type Vector is needed"
        if g is None:
            g = self.G[t][:, t]
        return super().forward(input, g)



class StaticGraphLinear(GraphLinear):
    def __init__(self, *args, bias: bool = True, num_nodes: int = None, graph_influence: Union[torch.Tensor, Parameter] = None,
                 learn_influence: bool = False, node_types: torch.Tensor = None, weights_per_type: bool = False, **kwargs):
        """
        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param num_nodes: Number of nodes.
        :param graph_influence: Graph Influence Matrix
        :param learn_influence: If set to ``False``, the layer will not learn an the Graph Influence Matrix.
        :param node_types: List of Type for each node. All nodes of same type will share weights.
                Default: All nodes have unique types.
        :param weights_per_type: If set to ``False``, the layer will not learn weights for each node type.
        :param bias: If set to ``False``, the layer will not learn an additive bias.
        """
        super().__init__(*args)

        self.learn_influence = learn_influence

        if graph_influence is not None:
            assert num_nodes == graph_influence.shape[0] or num_nodes is None, 'Number of Nodes or Graph Influence Matrix has to be given.'
            num_nodes = graph_influence.shape[0]
            if type(graph_influence) is Parameter:
                assert learn_influence, "Graph Influence Matrix is a Parameter, therefore it must be learnable."
                self.G = graph_influence
            elif learn_influence:
                self.G = Parameter(graph_influence)
            else:
                self.register_buffer('G', graph_influence)
        else:
            assert num_nodes, 'Number of Nodes or Graph Influence Matrix has to be given.'
            eye_influence = torch.eye(num_nodes, num_nodes)
            if learn_influence:
                self.G = Parameter(eye_influence)
            else:
                self.register_buffer('G', eye_influence)

        if weights_per_type and node_types is None:
            node_types = torch.tensor([i for i in range(num_nodes)])
        if node_types is not None:
            num_node_types = node_types.max() + 1
            self.weight = Parameter(torch.Tensor(num_node_types, self.out_features, self.in_features))
            self.mm = gmm
            self.node_type_index = node_types
        else:
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.mm = torch.matmul
            self.node_type_index = None

        if bias:
            if node_types is not None:
                self.bias = Parameter(torch.Tensor(num_node_types, self.out_features))
            else:
                self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


# class BN(Module):
#     def __init__(self, num_nodes, num_features):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.num_features = num_features
#         self.bn = BatchNorm1d(num_nodes * num_features)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.bn(x.view(-1, self.num_nodes * self.num_features)).view(-1, self.num_nodes, self.num_features)

# class LinearX(Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return input

