from typing import List
from torch import nn
from binsim.neural.utils import get_activation_by_name
import torch


class Dense(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layers=2,
                 activation: str = 'relu',
                 bias=False,
                 hidden_layers: List = None,
                 device=None,
                 dtype=None):
        """
        A simple implementation of the multi-layer perceptron.
        :param in_dim: the size of input tensor.
        :param out_dim: the size of output tensor.
        :param layers: the layer number of perceptron.
        :param activation: the activation function.
        :param bias: whether to use bias, whether the Linear layers should use bias.
        :param hidden_layers: a list of hidden layer sizes in order, if specified, layers is ignored
        """
        super(Dense, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        # initialize hidden layers
        activation_layer = get_activation_by_name(activation)
        layer_list = []
        if hidden_layers is None:
            hidden_layers = [out_dim] * layers
        last_layer_size = in_dim
        for layer_size in hidden_layers:
            layer_list.append(nn.Linear(last_layer_size, layer_size, bias=bias, **factory_kwargs))
            layer_list.append(activation_layer)
            last_layer_size = layer_size
        layer_list.append(nn.Linear(last_layer_size, out_dim, bias=bias, **factory_kwargs))
        self.layer = nn.Sequential(*layer_list)

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward propagation.
        :param inputs: A tensor of shape (N_1, N_2, ..., N_k, in_dim).
        :return:  A tensor of shape (N_1, N_2, ..., N_k, out_dim).
        """
        return self.layer(inputs)
