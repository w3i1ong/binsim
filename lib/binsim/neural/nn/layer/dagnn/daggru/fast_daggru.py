import torch
from torch.cuda.streams import Stream, Event
from ..dagrnn_ops import fused_gru_partial_forward, message_passing_forward, fused_gru_partial_backward

def calculate_w_x(node_batch_list, node_base_list, start, end, features, wx, weights_x, events):
    for j in range(start, end):
        batch_nodes = node_batch_list[j]
        node_num, node_base = batch_nodes.size(0), node_base_list[j]
        batch_features, batch_wx = features[node_base: node_base+node_num], wx[node_base: node_base+node_num]
        torch.matmul(batch_features, weights_x, out=batch_wx)
        events[j].record()

def gru_forward(node_batch_fwd, node_base_fwd, edge_batch_fwd, edge_batch_index_fwd,
                start, end,
                last_hidden, hidden, w_x, w_h, in_degrees,
                weights_h, biases_fwd,
                events,
                divide_degree=True, message_passing=True):
    current_stream = torch.cuda.current_stream(weights_h.device)
    batch_size, hidden_dim = w_h.shape[0], w_h.shape[1] // 3
    for j in range(start, end):
        node_batch = node_batch_fwd[j]
        node_num, node_base = node_batch.size(0), node_base_fwd[j]
        # divide degree
        if divide_degree and j != 0:
            last_hidden[node_base: node_base + node_num] /= in_degrees[node_base: node_base + node_num].unsqueeze(1)
        # calculate wh
        batch_lh, batch_wh = last_hidden[node_base: node_base + node_num], w_h[node_base: node_base + node_num]
        # invoke fused gru cell kernel
        torch.matmul(batch_lh, weights_h, out=batch_wh)
        current_stream.wait_event(events[j])
        fused_gru_partial_forward(last_hidden, hidden,
                                  w_x.view(-1, 3, hidden_dim), w_h.view(-1, 3, hidden_dim), biases_fwd.view(3, hidden_dim),
                                  node_base, node_num)
        if j < len(edge_batch_fwd) and message_passing:
            # invoke another kernel
            edges, edge_index = edge_batch_fwd[j], edge_batch_index_fwd[j]
            message_passing_forward(last_hidden, hidden, edges.long(), edge_index.long())


def daggru_forward(features, weights_x, weights_h, biases,
                   props_info: 'FastDAGGRUPropInfo',
                   divide_degree=True, message_passing=True,
                   layers=1, bidirectional=False, step=10):
    batch_size, in_dim, hidden_dim = len(features), features.size(1), biases.size(-1) // 3
    node_batch_num = len(props_info.node_batch_fwd)
    factory_kwargs = {'dtype': features.dtype, 'device': features.device}
    direction_num = 1 + bidirectional
    default_stream = torch.cuda.current_stream(features.device)
    # accelerate computation with streams
    # #0 for forward, #1 for backward
    streams = [(Stream(), Stream()) for _ in range(direction_num)]
    # temp variables
    w_x = torch.zeros((layers, batch_size, direction_num, hidden_dim * 3), **factory_kwargs)
    last_hidden = torch.zeros((layers, batch_size, direction_num, hidden_dim), **factory_kwargs)
    w_h, hidden = torch.zeros_like(w_x), torch.zeros_like(last_hidden)
    # input for the first layer is features
    layer_input = features
    params = [
        (props_info.node_batch_fwd, props_info.node_base_fwd, props_info.edge_batch_fwd,
            props_info.edge_batch_index_fwd, props_info.in_degrees),
        (props_info.node_batch_bck, props_info.node_base_bck, props_info.edge_batch_bck,
            props_info.edge_batch_index_bck, props_info.out_degrees),
    ]
    for layer_index in range(layers):
        events = [[Event() for _ in range(node_batch_num)] for _ in range(direction_num)]
        # get weights_x and weights_h of current layer
        if layer_index:
            weights_x_row_num = hidden_dim * direction_num
            weights_x_row_start = in_dim + weights_x_row_num * (layer_index - 1)
            layer_weights_x = weights_x[weights_x_row_start:weights_x_row_start+weights_x_row_num]
        else:
            layer_weights_x = weights_x[:in_dim]
        layer_weights_h = weights_h[layer_index]
        # mm_stream should wait for default stream and gru_stream, as the layer_input may have not been prepared.
        for (mm_stream, gru_stream) in streams:
            mm_stream.wait_stream(gru_stream)
            mm_stream.wait_stream(default_stream)

        for i in range(0, node_batch_num, step):
            start, end = i, min(i+step, node_batch_num)
            for direct in range(direction_num):
                node_batch_list, node_base_list, \
                    edge_batch_list, edge_batch_index_list, direction_in_degrees = params[direct]
                # forward direction
                with torch.cuda.stream(streams[direct][0]):
                    calculate_w_x(node_batch_list, node_base_list, start, end,
                                  features=layer_input, weights_x=layer_weights_x[:, direct],
                                  wx=w_x[layer_index,:,direct], events=events[direct])
                with torch.cuda.stream(streams[direct][1]):
                    gru_forward(node_batch_fwd=node_batch_list, node_base_fwd=node_base_list,
                                edge_batch_fwd=edge_batch_list, edge_batch_index_fwd=edge_batch_index_list,
                                start=start, end=end,
                                last_hidden=last_hidden[layer_index,:,direct], hidden=hidden[layer_index,:,direct],
                                w_x=w_x[layer_index,:,direct], w_h=w_h[layer_index,:,direct],
                                weights_h=layer_weights_h[:, direct], biases_fwd=biases[layer_index, direct],
                                in_degrees=direction_in_degrees.clip(1), events=events[direct],
                                divide_degree=divide_degree, message_passing=message_passing)
        layer_input = hidden[layer_index].view(batch_size, -1)
    # let current default stream wait two forward streams
    for _, gru_stream in streams:
        default_stream.wait_stream(gru_stream)
    return last_hidden, hidden, w_x, w_h

def gru_backward(node_batch_bck, node_base_bck, edge_batch_bck, edge_batch_index_bck, node_index,
                 grad_last_hidden, grad_output, grad_features, grad_weights_h, grad_weights_x,
                 last_hidden, features, w_x, w_h, weights_x, weights_h, biases, in_degrees,
                 divide_degree=True, message_passing=True):
    node_batch, node_base = node_batch_bck[node_index], node_base_bck[node_index]
    node_range = slice(node_base, node_base + node_batch.size(0))
    node_batch_num, batch_size = len(node_batch_bck), w_x.shape[0]
    # gru partial backward
    fused_gru_partial_backward(grad_output, grad_last_hidden, last_hidden,
                               w_x.view(batch_size, 3, -1), w_h.view(batch_size, 3, -1),
                               biases.view(3, -1), node_base, node_batch.size(0))
    # backward for wx, wh
    grad_last_hidden[node_range].addmm_(w_h[node_range], weights_h.T)
    grad_weights_h.addmm_(last_hidden[node_range].T, w_h[node_range])
    grad_features[node_range].addmm_(w_x[node_range], weights_x.T)
    grad_weights_x.addmm_(features[node_range].T, w_x[node_range])
    # divide degree
    if divide_degree and node_index != node_batch_num - 1:
        grad_last_hidden[node_range] /= in_degrees[node_range].unsqueeze(1).clip(1)
    # message passing
    if node_index != len(node_batch_bck) - 1 and message_passing:
        edges, edge_index = edge_batch_bck[node_index], edge_batch_index_bck[node_index]
        message_passing_forward(grad_output, grad_last_hidden, edges.long(), edge_index.long())

def daggru_backward(grad_output, grad_features,
                    grad_weights_x, grad_weights_h, grad_biases,
                    last_hidden, hidden, w_x, w_h,
                    features, weights_x, weights_h, biases,
                    props_info, layer = 1, bidirectional = False,
                    divide_degree=True, message_passing=True):
    in_dim, hidden_dim = features.size(1), biases.size(2) // 3
    direction_num = 1 + bidirectional
    grad_last_hidden = torch.zeros_like(grad_output)

    streams = [Stream() for _ in range(direction_num)]
    current_stream = torch.cuda.current_stream(features.device)

    grad_cur_layer_hidden = grad_output
    for layer_index in range(layer - 1, -1, -1):
        # get GRU weights
        if layer_index:
            cur_row_num = hidden_dim * direction_num
            cur_row_start = in_dim + cur_row_num * (layer_index - 1)
            layer_weights_x = weights_x[cur_row_start: cur_row_start + cur_row_num]
            grad_layer_weights_x = grad_weights_x[cur_row_start: cur_row_start + cur_row_num]
            layer_input = hidden[layer_index-1].view(-1, direction_num*hidden_dim)
            grad_layer_input = torch.zeros(direction_num, len(features), cur_row_num, device=features.device, dtype=features.dtype)
        else:
            layer_weights_x, grad_layer_weights_x = weights_x[:in_dim], grad_weights_x[:in_dim]
            layer_input = features
            grad_layer_input = torch.zeros(direction_num, len(features), in_dim, device=features.device, dtype=features.dtype)


        layer_weights_h, grad_layer_weights_h = weights_h[layer_index], grad_weights_h[layer_index]
        layer_bias = biases[layer_index]
        grad_layer_bias = grad_biases[layer_index]
        layer_last_hidden = last_hidden[layer_index]
        layer_w_x, layer_w_h = w_x[layer_index], w_h[layer_index]

        for stream in streams:
            stream.wait_stream(current_stream)
        for i in range(0, len(props_info.node_batch_fwd)):
            with torch.cuda.stream(streams[0]):
                gru_backward(props_info.node_batch_bck, props_info.node_base_bck,
                             props_info.edge_batch_bck, props_info.edge_batch_index_bck, i,
                             grad_last_hidden[:,0], grad_cur_layer_hidden[:,0], grad_layer_input[0],
                             grad_layer_weights_h[:, 0], grad_layer_weights_x[:, 0],
                             layer_last_hidden[:, 0], layer_input,
                             layer_w_x[:, 0], layer_w_h[:, 0],
                             layer_weights_x[:, 0], layer_weights_h[:, 0], layer_bias[0],
                             props_info.in_degrees, divide_degree, message_passing)
            if bidirectional:
                with torch.cuda.stream(streams[1]):
                    gru_backward(props_info.node_batch_fwd, props_info.node_base_fwd,
                                 props_info.edge_batch_fwd, props_info.edge_batch_index_fwd, i,
                                 grad_last_hidden[:,1], grad_cur_layer_hidden[:,1], grad_layer_input[1],
                                 grad_layer_weights_h[:, 1], grad_layer_weights_x[:, 1],
                                 layer_last_hidden[:, 1], layer_input,
                                 layer_w_x[:, 1], layer_w_h[:, 1],
                                 layer_weights_x[:, 1], layer_weights_h[:, 1], layer_bias[1],
                                 props_info.out_degrees, divide_degree, message_passing)

        # let default stream wait two backward streams
        for stream in streams:
            current_stream.wait_stream(stream)

        grad_layer_bias += layer_w_x.sum(dim=0)
        grad_layer_input = grad_layer_input.sum(dim=0)
        if layer_index == 0:
            grad_features += grad_layer_input
        else:
            grad_cur_layer_hidden = grad_layer_input.view(-1, direction_num, hidden_dim)
