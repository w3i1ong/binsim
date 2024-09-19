import torch
from torch.cuda.streams import Stream, Event
from binsim.neural.nn.layer.dagnn.dagrnn_ops import message_passing_forward, fused_lstm_partial_forward, fused_lstm_partial_backward

def calculate_w_xh(node_batch_list, node_base_list, start, end, inputs, w_xh, weights, bias, events):
    for j in range(start, end):
        batch_nodes = node_batch_list[j]
        node_num, node_base = batch_nodes.size(0), node_base_list[j]
        index = slice(node_base, node_base + node_num)
        w_xh[index].addmm_(inputs[index], weights)
        w_xh[index] += bias[None]
        events[j].record()

def lstm_forward(node_batch_fwd, node_base_fwd, edge_batch_fwd, edge_batch_index_fwd,
                 start, end, last_cell, last_hidden, weights_h, cell, hidden, w_xh,
                 events):
    current_stream = torch.cuda.current_stream(hidden.device)
    for j in range(start, end):
        node_batch = node_batch_fwd[j]
        node_num, node_base = node_batch.size(0), node_base_fwd[j]
        index = slice(node_base, node_base + node_num)
        batch_lc = last_cell[index]
        batch_lh = last_hidden[index]
        batch_cell = cell[index]
        batch_hidden = hidden[index]

        batch_wh = batch_lh @ weights_h

        current_stream.wait_event(events[j])
        batch_xh = w_xh[index]
        batch_xh += batch_wh
        i, f, g, o = torch.chunk(batch_xh, 4, 1)
        # invoke fused gru cell kernel
        fused_lstm_partial_forward(batch_lc, batch_lh, batch_cell, batch_hidden, i, f, g, o)
        if j < len(edge_batch_fwd):
            # invoke another kernel
            edges, edge_index = edge_batch_fwd[j], edge_batch_index_fwd[j]
            message_passing_forward(last_hidden, hidden, edges.long(), edge_index.long())
            message_passing_forward(last_cell, cell, edges.long(), edge_index.long())

def treelstm_forward(features, weights_x, weights_h, biases, props_info: 'FastDAGGRUPropInfo',
                     layers=1, step=10):
    node_num, in_dim, hidden_dim = len(features), features.size(1), biases.size(-1) // 4
    node_batch_num = len(props_info.node_batch_fwd)
    factory_kwargs = {'dtype': features.dtype, 'device': features.device}
    default_stream = torch.cuda.current_stream(features.device)
    # accelerate computation with streams
    # #0 for forward, #1 for backward
    stream = Stream()
    # temp variables
    w_xh = torch.zeros((layers, node_num, hidden_dim * 4), **factory_kwargs)
    last_hidden = torch.zeros((layers, node_num, hidden_dim), **factory_kwargs)
    hidden = torch.zeros_like(last_hidden)
    cell, last_cell = torch.zeros_like(hidden), torch.zeros_like(hidden)
    # input for the first layer is features
    layer_input = features
    for layer_index in range(layers):
        events = [Event() for _ in range(node_batch_num)]
        # get weights_x and weights_h of current layer
        if layer_index:
            weights_x_row_start = in_dim + hidden_dim * (layer_index - 1)
            layer_weights_x = weights_x[weights_x_row_start:weights_x_row_start+hidden_dim]
        else:
            layer_weights_x = weights_x[:in_dim]
        layer_weights_h, layer_bias = weights_h[layer_index], biases[layer_index]
        layer_last_hidden, layer_last_cell = last_hidden[layer_index], last_cell[layer_index]
        layer_w_xh = w_xh[layer_index]
        layer_hidden, layer_cell = hidden[layer_index], cell[layer_index]

        for i in range(0, node_batch_num, step):
            start, end = i, min(i+step, node_batch_num)
            node_batch_list, node_base_list, edge_batch_list, edge_batch_index_list = \
                props_info.node_batch_fwd, props_info.node_base_fwd, props_info.edge_batch_fwd, \
                    props_info.edge_batch_index_fwd
            # forward direction
            stream.wait_stream(default_stream)
            with torch.cuda.stream(stream):
                calculate_w_xh(node_batch_list, node_base_list, start, end,
                              inputs=layer_input, weights=layer_weights_x, bias=layer_bias,
                              w_xh=layer_w_xh, events=events)
            lstm_forward(node_batch_fwd=node_batch_list, node_base_fwd=node_base_list,
                        edge_batch_fwd=edge_batch_list, edge_batch_index_fwd=edge_batch_index_list,
                        start=start, end=end,  weights_h=layer_weights_h,
                        last_hidden=layer_last_hidden, last_cell=layer_last_cell,
                        hidden=layer_hidden, cell=layer_cell,
                        w_xh=layer_w_xh, events=events)
        layer_input = layer_hidden.view(node_num, -1)
    return last_hidden, last_cell, hidden, cell, w_xh

class DAGGRUFunction(torch.autograd.Function):
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

def lstm_backward(node_batch_bck, node_base_bck, edge_batch_bck, edge_batch_index_bck, node_index,
                 grad_last_hidden, grad_last_cell, grad_hidden, grad_cell, grad_features, grad_weights_h, grad_weights_x,
                 last_hidden, last_cell, features, w_xh, weights_x, weights_h):
    node_batch, node_base = node_batch_bck[node_index], node_base_bck[node_index]
    node_range = slice(node_base, node_base + node_batch.size(0))
    # gru partial backward
    cur_w_xh = w_xh[node_range]
    i,f,g,o = torch.chunk(cur_w_xh, 4, 1)
    fused_lstm_partial_backward(grad_cell=grad_cell[node_range], grad_hidden=grad_hidden[node_range],
                                grad_last_cell=grad_last_cell[node_range], grad_last_hidden=grad_last_hidden[node_range],
                                last_cell=last_cell[node_range], last_hidden=last_hidden[node_range],
                                i=i, f=f, g=g, o=o)
    # backward for wx, wh
    grad_last_hidden[node_range].addmm_(cur_w_xh, weights_h.T)
    grad_weights_h.addmm_(last_hidden[node_range].T, cur_w_xh)
    grad_features[node_range].addmm_(cur_w_xh, weights_x.T)
    grad_weights_x.addmm_(features[node_range].T, cur_w_xh)

    # message passing
    if node_index != len(node_batch_bck) - 1:
        edges, edge_index = edge_batch_bck[node_index], edge_batch_index_bck[node_index]
        message_passing_forward(grad_hidden, grad_last_hidden, edges.long(), edge_index.long())
        message_passing_forward(grad_cell, grad_last_cell, edges.long(), edge_index.long())

def treelstm_backward(grad_output, grad_features,
                    grad_weights_x, grad_weights_h, grad_biases,
                    last_hidden, last_cell, hidden, w_xh,
                    features, weights_x, weights_h,
                    props_info, layer = 1):
    in_dim, hidden_dim = features.size(1), weights_h.size(-1) // 4
    grad_last_hidden = torch.zeros_like(grad_output)
    grad_last_cell = torch.zeros_like(grad_output)

    grad_cur_layer_hidden = grad_output
    grad_cur_layer_cell = torch.zeros_like(grad_output)
    for layer_index in range(layer - 1, -1, -1):
        # get GRU weights
        if layer_index:
            cur_row_start = in_dim + hidden_dim * (layer_index - 1)
            layer_weights_x = weights_x[cur_row_start: cur_row_start + hidden_dim]
            grad_layer_weights_x = grad_weights_x[cur_row_start: cur_row_start + hidden_dim]
            layer_input = hidden[layer_index-1].view(-1, hidden_dim)
            grad_layer_input = torch.zeros(len(features), hidden_dim, device=features.device, dtype=features.dtype)
        else:
            layer_weights_x, grad_layer_weights_x = weights_x[:in_dim], grad_weights_x[:in_dim]
            layer_input = features
            grad_layer_input = torch.zeros(len(features), in_dim, device=features.device, dtype=features.dtype)
        grad_cur_layer_cell *= 0

        layer_weights_h, grad_layer_weights_h = weights_h[layer_index], grad_weights_h[layer_index]
        grad_layer_bias = grad_biases[layer_index]
        layer_last_hidden = last_hidden[layer_index]
        layer_last_cell = last_cell[layer_index]
        layer_w_xh= w_xh[layer_index]

        for i in range(0, len(props_info.node_batch_fwd)):
            lstm_backward(props_info.node_batch_bck, props_info.node_base_bck,
                         props_info.edge_batch_bck, props_info.edge_batch_index_bck, i,
                         grad_last_hidden, grad_last_cell,
                         grad_cur_layer_hidden, grad_cur_layer_cell, grad_layer_input,
                         grad_layer_weights_h, grad_layer_weights_x,
                         layer_last_hidden, layer_last_cell,
                         layer_input, layer_w_xh,
                         layer_weights_x, layer_weights_h)

        grad_layer_bias += layer_w_xh.sum(dim=0)
        if layer_index == 0:
            grad_features += grad_layer_input
        else:
            grad_cur_layer_hidden = grad_layer_input.view(-1, hidden_dim)

