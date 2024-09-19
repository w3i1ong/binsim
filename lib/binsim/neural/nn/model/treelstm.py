import dgl, torch
import torch.nn as nn
from collections import defaultdict
from binsim.neural.nn.layer.dagnn.treelstm import FastTreeLSTM
from binsim.neural.nn.layer.dagnn.utils.utils import prepare_update_information_for_faster_forward
from binsim.neural.nn.base.model import GraphEmbeddingModelBase

class AsteriaDistance(nn.Module):
    def __init__(self, in_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Linear(2 * in_dim, 2, bias=False, **factory_kwargs)
        # note: the original paper claims they utilized BCELoss as the loss function
        # But as I am not familiar with BCELoss, I utilize cross-entropy, they are equivalent，
        self._loss = nn.CrossEntropyLoss()

    def forward(self, embeddings1, embeddings2) -> torch.Tensor:
        embeddings1, callee_num1 = embeddings1[:,:-1], embeddings1[:,-1]
        embeddings2, callee_num2 = embeddings2[:,:-1], embeddings2[:,-1]
        difference = torch.abs(embeddings1 - embeddings2)
        product = embeddings1 * embeddings2
        logit = self.W(torch.sigmoid(torch.cat([difference, product], dim=-1)))

        S = torch.exp(-torch.abs(callee_num1-callee_num2))
        return logit * S.reshape([-1,1])

    def distance(self, embeddings1, embeddings2) -> torch.Tensor:
        return torch.softmax(self(embeddings1, embeddings2), dim=1)[:, 0]

    def pairwise_distance(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        embeddings1 = torch.unsqueeze(embeddings1, dim=1)
        embeddings2 = torch.unsqueeze(embeddings2, dim=0)
        embeddings1, callee_num1 = embeddings1[:,:,:-1], embeddings1[:,:,-1]
        embeddings2, callee_num2 = embeddings2[:,:,:-1], embeddings2[:,:,-1]
        difference = torch.abs(embeddings1 - embeddings2)
        product = embeddings1 * embeddings2
        logit = self.W(torch.sigmoid(torch.cat([difference, product], dim=-1)))
        S = torch.exp(-torch.abs(callee_num1-callee_num2))
        return torch.softmax(logit * S.reshape([len(embeddings1),embeddings2.shape[1], 1]), dim=-1)[:, :, 0]

    def siamese_loss(self, embeddings1, embeddings2, labels):
        return self._loss(self(embeddings1, embeddings2), labels.long())


class ChildSumTreeLSTM(GraphEmbeddingModelBase):
    def __init__(self, distance_func, in_dim=16, use_fast=False,
                 vocab_size=200, out_dim=200, device=None, dtype=None):
        if isinstance(distance_func, nn.Module):
            distance_func = distance_func.to(device)
        super(ChildSumTreeLSTM, self).__init__(distance_func=distance_func)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.use_fast = use_fast
        # initialize token embedding
        self.embedding = nn.Embedding(vocab_size, self.in_dim, **factory_kwargs)
        if use_fast:
            self.cell = FastTreeLSTM(in_dim=in_dim, hidden_dim=out_dim, **factory_kwargs)
        else:
            self.cell = nn.LSTMCell(in_dim, out_dim, False, **factory_kwargs)

    @property
    def graphType(self):
        raise NotImplementedError("graphType has not been implemented for ChildSumTreeLSTM")

    @property
    def sampleDataset(self):
        raise NotImplementedError("sampleDataloader has not been implemented for ChildSumTreeLSTM")

    def _prepare_update_information_for_faster_forward(self, g: dgl.DGLGraph):
        edge_src, edge_dst = g.all_edges()
        edge_src, edge_dst = edge_src.cpu().numpy(), edge_dst.cpu().numpy()
        edges = list(zip(edge_src, edge_dst))
        edges.sort()
        adj_list = defaultdict(list)
        for (src, dst) in edges:
            adj_list[src].append(dst)

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
        return forward_update_info

    def generate_embedding(self, graph, inputs, callee_num, root_idx) -> torch.Tensor:
        if self.use_fast:
            embeds = self.embedding(inputs)
            return self.fast_generate_embedding(graph, embeds, callee_num, root_idx)
        else:
            return self.slow_generate_embedding(graph, inputs, callee_num, root_idx)

    def fast_generate_embedding(self, graph, inputs: torch.Tensor, callee_num: torch.Tensor, root_idx)\
            -> torch.Tensor:
        embeddings = self.cell(graph, inputs)
        function_embedding = embeddings[root_idx]
        return torch.cat([function_embedding, callee_num[:, None]], dim=1)

    def slow_generate_embedding(self, graph: dgl.DGLGraph, inputs: torch.Tensor, callee_num: torch.Tensor, root_idx)\
            -> torch.Tensor:
        # feed embedding
        embeds = self.embedding(inputs)
        h = torch.zeros([graph.number_of_nodes(), self.hidden_dim], device=graph.device)
        c = torch.zeros([graph.number_of_nodes(), self.hidden_dim], device=graph.device)
        last_node_h = torch.zeros_like(h)
        last_node_c = torch.zeros_like(c)
        # calculate propagation information
        forward_update_info = self._prepare_update_information_for_faster_forward(graph)
        # propagate
        for nodes, (src, dst) in forward_update_info:
            node_features = embeds[nodes]
            h_children = last_node_h[nodes]
            c_children = last_node_c[nodes]

            new_h, new_c = self.cell(node_features, (h_children, c_children))
            h.index_copy_(0, nodes, new_h)
            c.index_copy_(0, nodes, new_c)

            # message passing
            if len(src):
                message_h, message_c = h[src], c[src]
                last_node_h.index_add_(0, dst, message_h)
                last_node_c.index_add_(0, dst, message_c)
        function_embedding = h[root_idx]
        return torch.cat([function_embedding, callee_num[:, None]], dim=1)


    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self.embedding.parameters())
        return {'total': total, 'token': token}
