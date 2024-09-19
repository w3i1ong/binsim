import dgl
import torch
from torch import nn
from binsim.neural.nn.base.model import GraphMatchingModelBase
from typing import Tuple, Any
import dgl.function as fn
from dgl.udf import EdgeBatch
from binsim.neural.nn.distance import PairwiseEuclidianDistance,\
    EuclidianDistance


class GraphMatchingLayer(nn.Module):
    def __init__(self, n_fea_dim, e_fea_dim, distance_func: nn.Module, pairwise_distance_func: nn.Module,
                 dtype=None, device=None):
        super().__init__()
        factory_kwargs = {'dtype': dtype, 'device': device}
        self.fn_message = nn.Sequential(
            nn.Linear(n_fea_dim * 2 + e_fea_dim, n_fea_dim, True, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(n_fea_dim, n_fea_dim, True, **factory_kwargs),
        )
        self.fn_update = nn.GRUCell(n_fea_dim * 2, n_fea_dim, **factory_kwargs)
        self.pairwise_distance_func = pairwise_distance_func
        self.distance_func = distance_func

    def _message_func(self, edges: EdgeBatch):
        concat_tensor = torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=-1)
        return {'m': self.fn_message(concat_tensor)}

    def _pass_message(self, graph: dgl.DGLGraph, n_features: torch.Tensor, e_features: torch.Tensor):
        graph.ndata['h'] = n_features
        graph.edata['e'] = e_features
        graph.update_all(self._message_func, fn.sum('m', 'm'))
        return graph.ndata.pop('m')

    def _cross_attention(self, graph: dgl.DGLGraph, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes_x = list(graph.batch_num_nodes().numpy())
        chunks = torch.split(node_features, num_nodes_x)

        chunks_after_attention = []
        for pair_idx in range(0, len(chunks), 2):
            chunk_x_after_attention, chunk_y_after_attention = \
                self._cross_attention_one_pair(chunks[pair_idx], chunks[pair_idx + 1])
            chunks_after_attention.append(chunk_x_after_attention)
            chunks_after_attention.append(chunk_y_after_attention)

        return torch.cat(chunks_after_attention, dim=0)

    def _cross_attention_one_pair(self, n_features_x: torch.Tensor, n_features_y: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        pairwise_distance = self.pairwise_distance_func(n_features_x, n_features_y)
        weights_x = torch.softmax(pairwise_distance, dim=1)
        weights_y = torch.transpose(torch.softmax(pairwise_distance, dim=0), 1, 0)
        return n_features_x - weights_x @ n_features_y, n_features_y - weights_y @ n_features_x

    def forward(self, graph: dgl.DGLGraph, node_features: torch.Tensor, edge_features: torch.Tensor):
        m = self._pass_message(graph, node_features, edge_features)
        u = self._cross_attention(graph, node_features)

        node_features = self.fn_update(torch.cat([m, u], dim=1), node_features)
        return node_features


class GraphMatchingPoolingLayer(nn.Module):
    def __init__(self, embed_dim, dtype=None, device=None):
        super().__init__()
        factory_kwargs = {'dtype': dtype, 'device': device}
        self.mlp_gate = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.mlp = nn.Linear(embed_dim, embed_dim, True, **factory_kwargs)
        self.mlp_G = nn.Linear(embed_dim, embed_dim, True, **factory_kwargs)

    def forward(self, graph, features) -> torch.Tensor:
        features = self.mlp(features) * torch.sigmoid(self.mlp_gate(features))
        graph.ndata['h'] = features
        graph_embedding = dgl.sum_nodes(graph, 'h')
        graph.ndata.pop('h')
        return self.mlp_G(graph_embedding)


class GraphMatchingNet(GraphMatchingModelBase):
    def __init__(self, out_dim, vocab_size, gmn_layers=1, dtype=None, device=None):
        super().__init__(out_dim=out_dim)
        factory_kwargs = {'dtype': dtype, 'device': device}
        self.ins_embedding = nn.Embedding(vocab_size, out_dim, **factory_kwargs)
        self.ins_embedding.requires_grad_(True)
        self.pooling_layer = GraphMatchingPoolingLayer(out_dim, **factory_kwargs)

        self._pairwise_distance = PairwiseEuclidianDistance()
        self._distance_func = EuclidianDistance()
        self.gmn_layers = nn.ModuleList(
            [GraphMatchingLayer(out_dim, out_dim, self._distance_func, self._pairwise_distance,
                                **factory_kwargs) for _ in range(gmn_layers)])
        self.gamma = nn.Parameter(torch.tensor(1.0, **factory_kwargs), requires_grad=False)

    def forward(self, samples: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor], labels, sample_ids=None):
        graph, basic_block, bb_length = samples
        edge_features = torch.ones([graph.num_edges(), self._out_dim], device=basic_block.device)
        node_features = (basic_block, bb_length)
        return super().forward((graph, node_features, edge_features), labels, sample_ids)

    def _generate_match_embedding(self, graph: dgl.DGLGraph, node_features: Any,
                                  edge_features: torch.Tensor) -> torch.Tensor:
        # As many indexed in basic_block are just padding index, which may result in a large memory usage.
        # So I flatten the index sequence, and then use index_add to sum up the embeddings of the same basic block.
        # I believe this operation can save memory usage.
        basic_block, bb_length = node_features
        node_features = self._generate_basic_block_embedding(graph, basic_block, bb_length)
        for layer in self.gmn_layers:
            node_features = layer(graph, node_features, edge_features)
        return self.pooling_layer(graph, node_features)

    def siamese_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, sample_ids) -> torch.Tensor:
        embeddings = embeddings.view([len(embeddings) // 2, 2, -1])
        return torch.mean(
            torch.relu(self.gamma - labels * (1 - self._distance_func(embeddings[:, 0], embeddings[:, 1]))))

    def similarity(self, samples) -> torch.Tensor:
        graph, basic_block, length = samples
        edge_features = torch.ones([graph.num_edges(), self._out_dim], device=basic_block.device)
        node_features = (basic_block, length)
        embeddings = self._generate_match_embedding(graph, node_features, edge_features)
        embeddings = embeddings.view([len(embeddings) // 2, 2, -1])
        # notice: we use negative distance to represent similarity to make sure the larger the similarity, the more
        # similar the two samples are.
        return - self._distance_func(embeddings[:, 0], embeddings[:, 1])

    def generate_pairwise_match_embedding(self, x_samples, y_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_x, basic_block_x, bb_length_x = x_samples
        graph_y, basic_block_y, bb_length_y = y_samples
        batch_size_x = graph_x.batch_size
        batch_size_y = graph_y.batch_size
        node_features_x = self._generate_basic_block_embedding(graph_x, basic_block_x, bb_length_x)
        node_features_y = self._generate_basic_block_embedding(graph_y, basic_block_y, bb_length_y)
        node_numbers_x = list(graph_x.batch_num_nodes().numpy())
        node_numbers_y = list(graph_y.batch_num_nodes().numpy())

        graphs_x = dgl.unbatch(graph_x)
        graphs_y = dgl.unbatch(graph_y)
        node_features_chunk_x = torch.split(node_features_x, node_numbers_x)
        node_features_chunk_y = torch.split(node_features_y, node_numbers_y)

        graphs = []
        node_features_chunks = []

        for graph_x, chunk_x in zip(graphs_x, node_features_chunk_x):
            for graph_y, chunk_y in zip(graphs_y, node_features_chunk_y):
                graphs.extend([graph_x, graph_y])
                node_features_chunks.extend([chunk_x, chunk_y])
        graph_x = dgl.batch(graphs)
        node_features = torch.cat(node_features_chunks, dim=0)
        edge_features = torch.ones([graph_x.num_edges(), self._out_dim], device=node_features.device)
        for layer in self.gmn_layers:
            node_features = layer(graph_x, node_features, edge_features)
        match_embeddings = self.pooling_layer(graph_x, node_features)
        match_embeddings = match_embeddings.reshape([len(match_embeddings) // 2, 2, -1])
        match_embeddings_x, match_embeddings_y = match_embeddings[:, 0], match_embeddings[:, 1]
        match_embeddings_x = match_embeddings_x.reshape([batch_size_x, batch_size_y, -1])
        match_embeddings_y = match_embeddings_y.reshape([batch_size_y, batch_size_x, -1])
        return match_embeddings_x, match_embeddings_y

    def _generate_basic_block_embedding(self, graph: dgl.DGLGraph, basic_block: torch.Tensor, bb_length: torch.Tensor):
        with torch.no_grad():
            basic_blocks = []
            basic_block_id = []
            for bb_idx, bb_length in enumerate(bb_length.detach().cpu().numpy()):
                basic_blocks.append(basic_block[bb_idx][:bb_length])
                basic_block_id.extend([bb_idx] * bb_length)
            basic_block_id = torch.tensor(basic_block_id, device=basic_block.device, dtype=torch.long)
            basic_block = torch.cat(basic_blocks, dim=0)
        basic_block = self.ins_embedding(basic_block)
        node_features = torch.zeros([graph.number_of_nodes(), self._out_dim], device=basic_block.device) \
            .index_add(dim=0, index=basic_block_id, source=basic_block)
        return node_features

    def pairwise_similarity(self, x_samples: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
                            y_samples: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_embeddings, y_embeddings = self.generate_pairwise_match_embedding(x_samples, y_samples)
        y_embeddings = y_embeddings.permute([1, 0, 2])
        return self._distance_func(x_embeddings, y_embeddings)
