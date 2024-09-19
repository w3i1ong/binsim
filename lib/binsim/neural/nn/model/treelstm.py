import dgl
import torch
import torch.nn as nn
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from collections import defaultdict

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
    def __init__(self, in_dim=16, vocab_size=200, out_dim=200, sample_format=None, device=None, dtype=None):
        super(ChildSumTreeLSTM, self).__init__(sample_format=sample_format)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        # initialize token embedding
        self.embedding = nn.Embedding(vocab_size, self.in_dim, **factory_kwargs)
        self.cell = nn.LSTMCell(in_dim, out_dim, False, **factory_kwargs)
        self.distance_func = AsteriaDistance(out_dim, **factory_kwargs)

    @property
    def graphType(self):
        raise NotImplementedError("graphType has not been implemented for ChildSumTreeLSTM")

    @property
    def pairDataset(self):
        raise NotImplementedError("pairDataloader has not been implemented for ChildSumTreeLSTM")

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

    def generate_embedding(self, graph: dgl.DGLGraph, inputs: torch.Tensor, callee_num: torch.Tensor, root_idx)\
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
        return torch.cat([function_embedding, callee_num.reshape([-1,1])], dim=1)

    def siamese_loss(self, samples: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        samples = samples.reshape([labels.shape[0], 2, -1])
        return self.distance_func.siamese_loss(samples[:, 0], samples[:, 1], labels)

    def triplet_loss(self, anchors: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Triplet loss is not implemented for TreeLSTM!")

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.reshape([samples.shape[0] // 2, 2, -1])
        return self.distance_func.distance(samples[:, 0], samples[:, 1])

    def similarity_between_original(self, samples):
        embeddings = self.generate_embedding(*samples)
        return self.similarity(embeddings)

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.distance_func.pairwise_distance(embeddings1=x, embeddings2=y)

    def pairwise_similarity_between_original(self, samples_x, samples_y):
        embeddings_x, embeddings_y = self.generate_embedding(*samples_x), self.generate_embedding(*samples_y)
        return self.pairwise_similarity(embeddings_x, embeddings_y)

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self.embedding.parameters())
        return {'total': total, 'token': token}
