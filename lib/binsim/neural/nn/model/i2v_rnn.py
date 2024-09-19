import dgl
import torch
from dgl import sum_nodes
from torch.nn import Linear, GRU
from torch.nn.utils.rnn import pack_padded_sequence
from binsim.neural.lm import Ins2vec
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from binsim.neural.nn.layer import Structure2vec, DAGGRU


class I2vRNN(GraphEmbeddingModelBase):
    def __init__(self, out_dim,
                 ins2vec: Ins2vec,
                 fixed_length=150,
                 use_dag=False,
                 gnn_layers=2,
                 bidirectional=False,
                 sample_format=None,
                 device=None,
                 dtype=None):
        """
        An implementation of [I2vRNN](https://ruoyuwang.me/bar2019/pdfs/bar2019-paper20.pdf)
        :param out_dim: The dimension size of final graph embedding.
        :param ins2vec: This argument provides the embedding matrix for the instructions. If a string is provided, it is
        assumed to be a path to a pickle file which contains the embedding matrix. Otherwise, it is assumed to be the
        embedding matrix.
        :param fixed_length: The maximum length of instruction sequences.
            If this argument is set to  None, this model will use variable length RNN.
            If this argument is set to a number, this model will use fixed length RNN, and the length of each instruction
            sequence will be truncated to it.
        :param gnn_layers: The number of GNN layers.
        :param bidirectional: Whether to use bidirectional RNN.
        """
        super(I2vRNN, self).__init__(sample_format=sample_format)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.bidirectional = bidirectional
        if isinstance(ins2vec, str):
            ins2vec = Ins2vec.load(ins2vec)
        self.ins2vec = ins2vec.as_torch_model(freeze=True).to(device)
        self.gru = GRU(ins2vec.embed_dim,
                       out_dim // (bidirectional + 1),
                       num_layers=2,
                       batch_first=True,
                       bidirectional=bidirectional,
                       **factory_kwargs)
        self._use_dag = use_dag
        if not self._use_dag:
            self.layer = Structure2vec(out_dim, out_dim, dense_layers=1, iteration_round=gnn_layers, **factory_kwargs)
        else:
            self.layer = DAGGRU(out_dim, out_dim, layer_num=gnn_layers, bidirectional=False, **factory_kwargs)

        self.linear = Linear(out_dim,
                             out_dim,
                             bias=False,
                             **factory_kwargs)
        self.fixed_length = fixed_length
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.margin = 0.5

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.TokenCFG

    @property
    def pairDataset(self):
        from binsim.neural.utils.data import InsStrCFGSamplePairDataset
        return InsStrCFGSamplePairDataset

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data import InsStrCFGSampleDataset
        return InsStrCFGSampleDataset

    def siamese_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        labels = torch.mul(torch.sub(labels, 0.5), 2)
        samples = embeddings.view([len(embeddings) // 2, 2, -1])
        return self.loss_func(labels, torch.cosine_similarity(samples[:, 0], samples[:, 1]))

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view([len(samples) // 2, 2, -1])
        return 1 - torch.cosine_similarity(samples[:, 0], samples[:, 1])

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + torch.tensor(1e-08, device=x.device))
        y = y / torch.sqrt(torch.sum(y ** 2, dim=1, keepdim=True) + torch.tensor(1e-08, device=x.device))
        return 1 - x @ y.T

    def generate_embedding(self, graph: dgl.DGLGraph, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        :param graph: The CFG to be embedded. This graph should be a batched graph.
        :param features: The instruction sequences of basic blocks in CFG. The shape of this tensor is (num_of_nodes,
        max_length_of_instruction_sequences).
        :param lengths: The length of each instruction sequence. The shape of this tensor is (num_of_nodes,).
        :return:
        """

        if self.fixed_length is not None and features.shape[1] >= self.fixed_length:
            features = features[:, :self.fixed_length]
            lengths = torch.clip(lengths, 0, self.fixed_length)
        lengths = lengths.cpu()
        lengths, idx = torch.sort(lengths, descending=True)
        old_idx = torch.argsort(idx)

        features = features[idx]
        features = self.ins2vec(features)
        features = pack_padded_sequence(features, lengths, batch_first=True)
        _, features = self.gru(features)
        features = torch.transpose(features, 0, 1)[old_idx]

        if self.bidirectional:
            features = features[:, -2:, :]
            features = features.reshape(features.shape[0], -1)
        else:
            features = features[:, -1, :]

        if self._use_dag:
            assert 'nodeId' in graph.ndata
            features = features[graph.ndata['nodeId']]

        new_features = self.layer(graph, features)
        graph.ndata['x'] = new_features
        res = self.linear(sum_nodes(graph, 'x'))
        graph.ndata.pop('x')
        return res

    def triplet_loss(self, anchors: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Triplet loss is not implemented for I2vRNN.")

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self.ins2vec.parameters())
        return {'total': total, 'token': token, 'model': total - token}
