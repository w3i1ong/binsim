from binsim.neural.lm import Ins2vec
import dgl
import torch
from dgl import sum_nodes
from torch.nn import Linear
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from binsim.neural.nn.layer import Structure2vec


class I2vAtt(GraphEmbeddingModelBase):
    def __init__(self, out_dim: int,
                 ins2vec: Ins2vec,
                 fixed_length=150,
                 use_mask=False,
                 sampler=None,
                 device=None,
                 dtype=None):
        """
        An implementation of [I2vAtt](https://ruoyuwang.me/bar2019/pdfs/bar2019-paper20.pdf)
        :param out_dim: The dimension size of final graph embedding.
        :param ins2vec: This argument provides the embedding matrix for the instructions. If a string is provided, it is
        assumed to be a path to a pickle file which contains the embedding matrix. Otherwise, it is assumed to be the
        embedding matrix.
        :param fixed_length: The maximum length of instruction sequences. If the length of a instruction sequence exceeds
        this value, the sequences will be truncated.
        """
        super(I2vAtt, self).__init__(sample_format=sampler)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # initialize embedding layer
        self.ins2vec = ins2vec.as_torch_model(freeze=True)

        self.position_weights = torch.nn.Parameter(torch.normal(0, 1, (1, fixed_length, 1), **factory_kwargs),
                                                   requires_grad=True)
        self.layer = Structure2vec(self.ins2vec.embedding_dim, out_dim, **factory_kwargs)
        self.linear = Linear(out_dim, out_dim, bias=False, **factory_kwargs)
        self.fixed_length = fixed_length

        range_matrix = torch.arange(fixed_length).unsqueeze(0).repeat(fixed_length, 1)
        self.mask = torch.lt(range_matrix, range_matrix.T + 1).to(dtype)
        self.use_mask = use_mask
        self.loss_func = torch.nn.MSELoss()

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.TokenCFG

    @property
    def pairDataset(self):
        from binsim.neural.utils.data import TokenCFGDataForm
        return TokenCFGDataForm

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data import TokenDAGSampleDataset
        return TokenDAGSampleDataset

    def generate_embedding(self, graph: dgl.DGLGraph, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        :param graph: The CFG to be embedded. This graph should be a batched graph.
        :param features: The instruction sequences of basic blocks in CFG. The shape of this tensor is (num_of_nodes,
        max_length_of_instruction_sequences).
        :param lengths: The length of each instruction sequence. The shape of this tensor is (num_of_nodes,).
        :return:
        """
        # process features to make sure the length of each instruction sequence is fixed_length
        if features.size(1) >= self.fixed_length:
            features = features[:, :self.fixed_length]
        else:
            paddings = torch.zeros([features.shape[0], self.fixed_length - features.shape[1]],
                                   device=features.device,
                                   dtype=features.dtype)
            features = torch.concat((features, paddings), dim=1)

        features = self.ins2vec(features)
        if self.use_mask:
            lengths = torch.clamp(lengths, max=self.fixed_length)
            mask = self.mask.to(graph.device)
            mask = torch.unsqueeze(mask, 2)[lengths]
            features = features * mask

        features = torch.sum(self.position_weights * features, 1)
        features = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-10)
        new_features = self.layer(graph, features)
        graph.ndata['x'] = new_features
        res = self.linear(sum_nodes(graph, 'x'))
        graph.ndata.pop('x')
        return res

    def siamese_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, sample_ids:torch.Tensor) -> torch.Tensor:
        labels = torch.mul(torch.sub(labels, 0.5), 2)
        return self.loss_func(self.similarity(embeddings), labels)

    def triplet_loss(self, anchors: torch.Tensor, positive:torch.Tensor, negative:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Triplet loss is not implemented for I2vAtt!")

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + torch.tensor(1e-08, device=x.device))
        y = y / torch.sqrt(torch.sum(y**2, dim=1, keepdim=True) + torch.tensor(1e-08, device=x.device))
        return x @ y.T

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view([len(samples) // 2, 2, -1])
        return torch.cosine_similarity(samples[:, 0], samples[:, 1])

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self.ins2vec.parameters())
        return {'total': total, 'token': token, 'model': total - token}
