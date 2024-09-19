import dgl
import torch
from torch import nn
from collections import defaultdict
from .fast_daggru import daggru_forward, daggru_backward
from ..utils import FastDAGGRUPropInfo, prepare_update_information_for_faster_forward

class DAGGRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights_x, weights_h, biases,
                props_info, layer_num, bidirectional,
                divide_in_degree=True, message_passing=True):
        last_hidden, hidden, wx, wh = daggru_forward(features, weights_x, weights_h, biases,
                                                     props_info, layers=layer_num, bidirectional=bidirectional,
                                                     divide_degree=divide_in_degree, message_passing=message_passing)
        ctx.save_for_backward(last_hidden, hidden, wx, wh, features, weights_x, weights_h, biases)
        ctx.extra_info = (props_info, bidirectional, layer_num, divide_in_degree, message_passing)
        return hidden[-1].flatten(1,)

    @staticmethod
    def backward(ctx, grad_hidden):
        props_info, bidirectional, layer_num, divide_in_degree, message_passing = ctx.extra_info
        last_hidden, hidden, wx, wh, features, weights_x, weights_h, biases = ctx.saved_tensors

        grad_features = torch.zeros_like(features)
        grad_weights_x, grad_weights_h = torch.zeros_like(weights_x), torch.zeros_like(weights_h)
        grad_biases = torch.zeros_like(biases)
        gru_hidden = grad_hidden.view(len(grad_hidden), 1 + bidirectional, -1)

        daggru_backward(gru_hidden, grad_features,
                        grad_weights_x=grad_weights_x, grad_weights_h=grad_weights_h, grad_biases=grad_biases,
                        last_hidden=last_hidden, hidden=hidden, w_x=wx, w_h=wh,
                        features=features, weights_x=weights_x, weights_h=weights_h, biases=biases,
                        props_info=props_info,
                        layer=layer_num, bidirectional=bidirectional,
                        divide_degree=divide_in_degree,
                        message_passing=message_passing)
        return grad_features, grad_weights_x, grad_weights_h, grad_biases, \
            None, None, None, None, None


class FastDAGGRU(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 layer_num=1,
                 bidirectional=False,
                 divide_in_degree=True,
                 message_passing=True,
                 device=None,
                 dtype=None):
        super(FastDAGGRU, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        direction_num = 1 + bidirectional
        assert hidden_dim % direction_num == 0, "hidden_dim should be divisible by 2 if bidirectional is True."
        self.hidden_dim = hidden_dim
        hidden_dim //= direction_num
        self.in_dim = in_dim
        self.layer_num = layer_num
        self.bidirectional = bidirectional
        self.divide_in_degree = divide_in_degree
        self.message_passing = message_passing


        direction_num = 1 + bidirectional

        weights_row_num = in_dim + (layer_num - 1) * hidden_dim * direction_num
        self.weights_x = nn.Parameter(torch.empty(weights_row_num, direction_num, hidden_dim*3, **factory_kwargs))
        self.weights_h = nn.Parameter(torch.empty(layer_num, hidden_dim, direction_num, hidden_dim*3, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(layer_num, direction_num, hidden_dim*3, **factory_kwargs))
        self.reset_parameters()


    def forward(self, prop_info:FastDAGGRUPropInfo, features: torch.Tensor):
        if isinstance(prop_info, list):
            prop_info = prepare_update_information_for_faster_forward(prop_info).to(features.device)
        h = torch.zeros_like(features).index_add_(0,prop_info.index_map,features)
        h = DAGGRUFunction.apply(h,
                                 self.weights_x, self.weights_h, self.bias,
                                 prop_info, self.layer_num, self.bidirectional,
                                 self.divide_in_degree, self.message_passing)
        return h[prop_info.index_map]

    def reset_parameters(self) -> None:
        from torch.nn import init
        import math
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        init.uniform_(self.weights_x, -stdv, stdv)
        init.uniform_(self.weights_h, -stdv, stdv)
        init.uniform_(self.bias, -stdv, stdv)

class DAGGRU(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 layer_num=1,
                 bidirectional=False,
                 divide_in_degree=True,
                 message_passing=True,
                 device=None,
                 dtype=None):
        super(DAGGRU, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert hidden_dim % (bidirectional + 1) == 0, "hidden_dim should be divisible by 2 if bidirectional is True."
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim // (bidirectional + 1)
        self.bidirectional = bidirectional
        self.divide_in_degree= divide_in_degree
        self.message_passing = message_passing
        if bidirectional:
            self.forward_cell = nn.ModuleList(
                [nn.GRUCell(in_dim, self.hidden_dim, **factory_kwargs)] + \
                [nn.GRUCell(self.hidden_dim * 2, self.hidden_dim, **factory_kwargs) for _ in range(layer_num - 1)])
            self.backward_cell = nn.ModuleList(
                [nn.GRUCell(in_dim, self.hidden_dim, **factory_kwargs)] + \
                [nn.GRUCell(self.hidden_dim * 2, self.hidden_dim, **factory_kwargs) for _ in range(layer_num - 1)])
        else:
            self.forward_cell = nn.ModuleList(
                [nn.GRUCell(in_dim, self.hidden_dim, **factory_kwargs)] + \
                [nn.GRUCell(self.hidden_dim, self.hidden_dim, **factory_kwargs) for _ in range(layer_num - 1)])

    def _prepare_update_information_for_faster_forward(self, g: dgl.DGLGraph):
        edge_src, edge_dst = g.all_edges()
        edge_src, edge_dst = edge_src.cpu().numpy(), edge_dst.cpu().numpy()

        adj_list, adj_list_rev = defaultdict(list), defaultdict(list)
        in_degree, out_degree = [0] * g.number_of_nodes(), [0] * g.number_of_nodes()
        for (src, dst) in zip(edge_src, edge_dst):
            adj_list[src].append(dst)
            adj_list_rev[dst].append(src)
            in_degree[dst] += 1
            out_degree[src] += 1

        update_order = list(dgl.topological_nodes_generator(g))
        forward_update_info = []
        for idx, node_batch in enumerate(update_order):
            edge_src, edge_dst = [], []
            for src in node_batch.numpy():
                edge_src.extend([src] * len(adj_list[src]))
                edge_dst.extend(adj_list[src])

            forward_update_info.append((node_batch.to(g.device),
                                        (torch.tensor(edge_src, device=g.device),
                                         torch.tensor(edge_dst, device=g.device))))

        back_update_info = None
        if self.bidirectional:
            update_order_backward = list(dgl.topological_nodes_generator(g, reverse=True))
            back_update_info = []
            for idx, node_batch in enumerate(update_order_backward):
                edge_src, edge_dst = [], []
                for src in node_batch.numpy():
                    edge_src.extend([src] * len(adj_list_rev[src]))
                    edge_dst.extend(adj_list_rev[src])
                back_update_info.append((node_batch.to(g.device),
                                         (torch.tensor(edge_src, device=g.device),
                                          torch.tensor(edge_dst, device=g.device))))

        return forward_update_info, torch.tensor(in_degree, device=g.device, dtype=torch.float).reshape([-1, 1]), \
            back_update_info, torch.tensor(out_degree, device=g.device, dtype=torch.float).reshape([-1, 1]),

    def forward(self, g: dgl.DGLGraph, features: torch.Tensor):
        """
        This is a faster implementation for DAGGRU(compared to slower_forward). dgl creates a temporary graph for each
            message passing step, which results in extremely slow performance for DAGGRU.
        :param g:
        :param features:
        :return:
        """
        forward_info, in_degree, backward_info, out_degree = self._prepare_update_information_for_faster_forward(g)
        # As the in_degree of first batch of nodes are 0, there is no need to calculate average for them
        h = features
        for layer_idx in range(self.layer_num):
            h_forward = self.__forward_one_layer(forward_info, in_degree, h, self.forward_cell[layer_idx])
            if self.bidirectional:
                h_bck = self.__forward_one_layer(backward_info, out_degree, h, self.backward_cell[layer_idx])
                h = torch.cat([h_forward, h_bck], dim=1)
            else:
                h = h_forward
        return h

    def __forward_one_layer(self, update_info, in_degree, features: torch.Tensor, gru_cell):
        last_hidden = torch.zeros([features.shape[0], self.hidden_dim], device=features.device)
        h: torch.Tensor = torch.zeros_like(last_hidden)
        first_batch_skipped = False
        for node_batch, (edge_src, edge_dst) in update_info:
            # get the hidden states and input
            node_last_hidden, node_x = last_hidden[node_batch], features[node_batch]
            # average
            if first_batch_skipped and self.divide_in_degree:
                node_last_hidden /= in_degree[node_batch]
            else:
                first_batch_skipped = True
            # apply gru cell
            node_h = gru_cell(node_x, hx=node_last_hidden)
            h.index_copy_(0, node_batch, node_h)

            if len(edge_src) and self.message_passing:
                message = h[edge_src]
                last_hidden.index_add_(0, edge_dst, message)
        return h
