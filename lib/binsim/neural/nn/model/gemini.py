import dgl
import torch
from dgl import sum_nodes
from torch.nn import Linear, MSELoss
from binsim.neural.nn.layer import Structure2vec
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from binsim.neural.nn.globals.siamese import SiameseSampleFormat
from binsim.neural.nn.layer import DAGGRU


class Gemini(GraphEmbeddingModelBase):
    def __init__(self, in_dim: int, out_dim: int, *,
                 gnn_layers=5,
                 use_dag=False,
                 sample_format: SiameseSampleFormat = None,
                 zero=False,
                 device=None, dtype=None):
        """
        An implementation of [Gemini](https://arxiv.org/abs/1708.06525).
        :param in_dim: The dimension size of node features in in_graph.
        :param out_dim: The dimension size of node features in out_graph.
        :param gnn_layers: The number of GNN layers.
        :param use_dag: If True, Gemini will use DAGGRU instead of Structure2vec.
        :param sample_format: Which kind of sample pair will be used in training.
        :param zero: Whether to use zero features.
        :param device: The device to use.
        :param dtype: The data type to use.
        """
        super(Gemini, self).__init__(sample_format=sample_format)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self._use_dag = use_dag
        if not use_dag:
            self.gcn_layer = Structure2vec(in_dim, out_dim, iteration_round=gnn_layers, **factory_kwargs)
        else:
            self.gcn_layer = DAGGRU(in_dim, out_dim, layer_num=gnn_layers, bidirectional=False, **factory_kwargs)
        self.linear = Linear(out_dim, out_dim, bias=False, **factory_kwargs)
        self.loss = MSELoss(reduction='mean')
        self.margin = 0.5
        self.zero = zero

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.ACFG

    @property
    def pairDataset(self):
        from binsim.neural.utils.data import ACFGSamplePairDataset
        return ACFGSamplePairDataset

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data import ACFGSampleDataset
        return ACFGSampleDataset

    def siamese_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, sample_ids) -> torch.Tensor:
        labels: torch.Tensor = torch.mul(torch.sub(labels, 0.5), 2)
        embeddings = embeddings.reshape([embeddings.shape[0] // 2, 2, -1])
        anchor, another = embeddings[:, 0], embeddings[:, 1]
        return self.loss(torch.cosine_similarity(anchor, another), labels)

    def generate_embedding(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of gemini.
        :param graph: batched graphs to generate graph embedding
        :param features: features of each node
        :return: The generated graph embeddings.
        """
        if getattr(self,"_use_dag", False):
            assert 'nodeId' in graph.ndata, "DAGGRU requires nodeId in graph.ndata."
            features = features[graph.ndata['nodeId']]
        if getattr(self,'zero', False):
            features = torch.ones_like(features)
        features = self.gcn_layer(graph, features)
        graph.ndata['x'] = features
        res = self.linear(sum_nodes(graph, 'x'))
        graph.ndata.pop('x')
        return res

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view([len(samples) // 2, 2, -1])
        return 1 - torch.cosine_similarity(samples[:, 0], samples[:, 1])

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cosine_similarity(x[:, None], y[None], dim=2)

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
