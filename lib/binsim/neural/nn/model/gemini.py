import dgl, torch
from dgl import sum_nodes
from torch.nn import Linear
from binsim.neural.nn.layer import Structure2vec, DAGGRU
from binsim.neural.nn.base.model import GraphEmbeddingModelBase


class Gemini(GraphEmbeddingModelBase):
    def __init__(self, in_dim: int, out_dim: int, *,
                 distance_func=None,
                 gnn_layers=5,
                 use_dag=False,
                 device=None, dtype=None):
        """
        An implementation of [Gemini](https://arxiv.org/abs/1708.06525).
        :param in_dim: The dimension size of node features in in_graph.
        :param out_dim: The dimension size of node features in out_graph.
        :param gnn_layers: The number of GNN layers.
        :param use_dag: If True, Gemini will use DAGGRU instead of Structure2vec.
        :param device: The device to use.
        :param dtype: The data type to use.
        """
        super(Gemini, self).__init__(distance_func=distance_func)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self._use_dag = use_dag
        if not use_dag:
            self.gcn_layer = Structure2vec(in_dim, out_dim, iteration_round=gnn_layers, **factory_kwargs)
        else:
            self.gcn_layer = DAGGRU(in_dim, out_dim, layer_num=gnn_layers, bidirectional=False, **factory_kwargs)
        self.linear = Linear(out_dim, out_dim, bias=False, **factory_kwargs)

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.ACFG

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data import ACFGSampleDataset
        return ACFGSampleDataset

    def generate_embedding(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of gemini.
        :param graph: batched graphs to generate graph embedding
        :param features: features of each node
        :return: The generated graph embeddings.
        """
        if self._use_dag:
            assert 'nodeId' in graph.ndata, "DAGGRU requires nodeId in graph.ndata."
            features = features[graph.ndata['nodeId']]
        features = self.gcn_layer(graph, features)
        graph.ndata['x'] = features
        res = self.linear(sum_nodes(graph, 'x'))
        graph.ndata.pop('x')
        return res

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
