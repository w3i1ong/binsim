import dgl
from torch import nn
import dgl.function as fn
import torch
from .dense import Dense
from binsim.neural.utils import get_activation_by_name


class Structure2vec(nn.Module):
    def __init__(self, in_dim,
                 out_dim=64,
                 iteration_round=2,
                 dense_layers=1,
                 dense_activation='relu',
                 activation='tanh',
                 dense_bias=False,
                 bias=False,
                 device=None,
                 dtype=None):
        """
        The Structure2vec used in (Gemini)[https://arxiv.org/abs/1708.06525].
        It can be described as the following formula:
            u_{i}^{0} = 0       (1)
            u_{i}^{j} = activation(linear(u_{i}^{j-1}) + dense(x_{i})) 0 < j \lt k      (2)
        :param in_dim: the dimension size fo node features(x_{i}) in input graph.
        :param out_dim: the dimension size fo node features(u_{i}^{k}) in output graph.
        :param iteration_round: the number of iteration rounds (k).
        :param dense_layers: The layer number of dense layers.
        :param dense_activation: The activation function used in dense layers.
        :param activation: activation function in equation (2).
        :param dense_bias: whether to use bias in dense layers.
        :param bias: Whether to use bias in linear layer of equation2.
        """
        super(Structure2vec, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear = nn.Linear(in_dim, out_dim, bias, **factory_kwargs)
        self.dense = Dense(out_dim,
                           out_dim=out_dim,
                           layers=dense_layers,
                           activation=dense_activation,
                           bias=dense_bias,
                           **factory_kwargs)
        self.activation = get_activation_by_name(activation)
        self.iteration_round = iteration_round
        self.out_dim = out_dim

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.
        :param graph: The input graph, in which no node features is provided.
        :param feature: Node features of shape (num_of_nodes in_dim).
        :return: The updated node features of shape (num_of_nodes out_dim).
        """
        if self.iteration_round == 0:
            return feature
        node_features = self.linear(feature)
        graph.ndata['u'] = torch.zeros([feature.size(0), self.out_dim], device=graph.device)
        for i in range(self.iteration_round):
            graph.update_all(fn.copy_u('u', 'm'), fn.sum('m', 'm'))
            graph.ndata['u'] = self.activation(node_features + self.dense(graph.ndata.pop('m')))
        return graph.ndata.pop('u')
