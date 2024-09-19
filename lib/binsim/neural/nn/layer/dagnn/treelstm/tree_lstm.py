import torch
from torch import nn
from .fast_tree_lstm import treelstm_forward, treelstm_backward
from ..utils import FastDAGGRUPropInfo, prepare_update_information_for_faster_forward


class TreeLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights_x, weights_h, biases, props_info, layer_num):
        last_hidden, last_cell, hidden, cell, wxh = treelstm_forward(features, weights_x,
                                                                     weights_h, biases, props_info,
                                                                     layers=layer_num)
        ctx.save_for_backward(last_hidden, last_cell, cell, hidden, wxh, features, weights_x, weights_h, biases)
        ctx.extra_info = (props_info, layer_num)
        return hidden[-1].flatten(1,)

    @staticmethod
    def backward(ctx, grad_hidden):
        props_info, layer_num = ctx.extra_info
        last_hidden, last_cell, cell, hidden, wxh, features, weights_x, weights_h, biases = ctx.saved_tensors

        grad_features = torch.zeros_like(features)
        grad_weights_x, grad_weights_h = torch.zeros_like(weights_x), torch.zeros_like(weights_h)
        grad_biases = torch.zeros_like(biases)
        gru_hidden = grad_hidden.view(len(grad_hidden), -1)

        treelstm_backward(gru_hidden, grad_features,
                        grad_weights_x=grad_weights_x, grad_weights_h=grad_weights_h, grad_biases=grad_biases,
                        last_hidden=last_hidden, last_cell=last_cell, hidden=hidden, w_xh=wxh,
                        features=features, weights_x=weights_x, weights_h=weights_h,
                        props_info=props_info,
                        layer=layer_num)
        return grad_features, grad_weights_x, grad_weights_h, grad_biases, \
            None, None, None

class FastTreeLSTM(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 layer_num=1,
                 device=None,
                 dtype=None):
        super(FastTreeLSTM, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.layer_num = layer_num

        weights_row_num = in_dim + (layer_num - 1) * hidden_dim
        self.weights_x = nn.Parameter(torch.empty(weights_row_num, hidden_dim*4, **factory_kwargs))
        self.weights_h = nn.Parameter(torch.empty(layer_num, hidden_dim, hidden_dim*4, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(layer_num, hidden_dim*4, **factory_kwargs))
        self.reset_parameters()


    def forward(self, prop_info:FastDAGGRUPropInfo, features: torch.Tensor):
        if isinstance(prop_info, list):
            prop_info = prepare_update_information_for_faster_forward(prop_info).to(features.device)
        h = torch.zeros_like(features).index_add_(0,prop_info.index_map,features)
        h = TreeLSTMFunction.apply(h, self.weights_x, self.weights_h, self.bias,
                                 prop_info, self.layer_num)
        return h[prop_info.index_map]

    def reset_parameters(self) -> None:
        from torch.nn import init
        import math
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        init.uniform_(self.weights_x, -stdv, stdv)
        init.uniform_(self.weights_h, -stdv, stdv)
        init.uniform_(self.bias, -stdv, stdv)
