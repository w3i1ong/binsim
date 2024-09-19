import dgl, torch
from binsim.neural.nn.layer import DAGGRU
from torch import nn
from torch.nn import GRU
from torch.nn.utils.rnn import pack_padded_sequence
from binsim.neural.nn.base.model import GraphEmbeddingModelBase, SiameseSampleFormat
from binsim.neural.nn.layer import InstructionEmbedding, Structure2vec
from binsim.neural.utils.data import BatchedInstruction, BatchedBBIndex

class AttentionWeights(nn.Module):
    def __init__(self, in_dim, layer_num=2, device=None, dtype=None):
        super(AttentionWeights, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        layers = []
        for i in range(layer_num - 1):
            layers.append(nn.Linear(in_dim, in_dim, **factory_kwargs))
            layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Linear(in_dim, 1, **factory_kwargs))
        layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class RCFG2Vec(GraphEmbeddingModelBase):
    def __init__(self, in_dim,
                 embed_dim,
                 vocab_size,
                 layer_num=1,
                 rnn_layers=2,
                 margin=0.5,
                 gru_bidirectional=True,
                 dag_gru_bidirectional=False,
                 ins_embed_model='normal',
                 use_dag=True,
                 sample_format=SiameseSampleFormat.Pair,
                 embedding_queue_size=250,
                 device=None,
                 dtype=None):
        """
        An implementation of [RCFG2Vec]()
        :param embed_dim: The dimension of the graph embedding.
        :param vocab_size: The size of vocabulary.
        :param layer_num: The number of DAGGRU layers.
        :param rnn_layers: The number of GRU layers.
        :param gru_bidirectional: Whether to use bidirectional GRU.
        :param dag_gru_bidirectional: Whether to use bidirectional DAGGRU.
        """
        super(RCFG2Vec, self).__init__(sample_format=sample_format)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # initialize the instruction embedding layer
        self._ins_embedding = InstructionEmbedding(vocab_size, in_dim, **factory_kwargs, mode=ins_embed_model)
        assert rnn_layers > 0, 'rnn_layers must be greater than 0'
        self.gru = GRU(in_dim,
                       embed_dim // (gru_bidirectional + 1),
                       batch_first=True,
                       num_layers=rnn_layers,
                       bidirectional=gru_bidirectional,
                       **factory_kwargs)
        self.rnn_bidirectional = gru_bidirectional
        self.dag_gru_bidirectional = dag_gru_bidirectional
        self._use_dag = use_dag
        if use_dag:
            self.dag_gru = DAGGRU(embed_dim, embed_dim, layer_num=layer_num, bidirectional=dag_gru_bidirectional,
                                  **factory_kwargs)
        else:
            self.dag_gru = Structure2vec(embed_dim, embed_dim, iteration_round=layer_num, **factory_kwargs)

        self.margin = margin
        match sample_format:
            case SiameseSampleFormat.SemiHardPair | SiameseSampleFormat.Pair:
                self.loss_func = nn.CosineEmbeddingLoss(margin=margin)
            case SiameseSampleFormat.InfoNCESamples:
                self.loss_func = nn.CrossEntropyLoss()

        self._embedding_queue_size = embedding_queue_size
        self._embedding_queue = torch.normal(0, 1, [embedding_queue_size, embed_dim], **factory_kwargs)
        self._embedding_queue_idx = 0


    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.InsCFG

    @property
    def pairDataset(self):
        from binsim.neural.utils.data.dataset import InsCFGSamplePairDataset
        return InsCFGSamplePairDataset

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data.dataset import InsCFGSampleDataset
        return InsCFGSampleDataset

    def generate_basic_block_embeddings(self, unique_ins_embedding: torch.Tensor,
                                        basic_block_ins_index: BatchedBBIndex):
        basic_block_embedding_chunks = []
        for bb_index_chunk, bb_length_chunk in basic_block_ins_index.bb_chunks:
            embedding_seq = unique_ins_embedding[bb_index_chunk]
            embedding_seq = pack_padded_sequence(embedding_seq, bb_length_chunk, batch_first=True)
            _, features = self.gru(embedding_seq)
            features = torch.transpose(features, 0, 1)
            basic_block_embedding_chunks.append(features)
        basic_block_embeddings = torch.cat(basic_block_embedding_chunks, dim=0)
        return basic_block_embeddings[basic_block_ins_index.original_index]

    def generate_embedding(self, graph: dgl.DGLGraph,
                           basic_block_ins_index: BatchedBBIndex,
                           batched_instructions: BatchedInstruction) -> torch.Tensor:
        # generate embedding for each basic block
        unique_instruction_embeddings = self._ins_embedding(batched_instructions)
        features = self.generate_basic_block_embeddings(unique_instruction_embeddings,
                                                        basic_block_ins_index)

        if self.rnn_bidirectional:
            features = features[:, -2:, :]
            features = features.reshape(features.shape[0], -1)
        else:
            features = features[:, -1, :]

        if getattr(self, '_use_dag', True):
            assert 'nodeId' in graph.ndata, "When using DAGGRU, the graph must have nodeId features."
            features = features[graph.ndata['nodeId']]

        # generate graph embedding
        features = self.dag_gru(graph, features)
        graph.ndata['h'] = features
        embeddings = dgl.mean_nodes(graph, 'h')
        return embeddings

    def siamese_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        labels = torch.mul(torch.sub(labels, 0.5), 2)
        embeddings = embeddings.reshape([len(embeddings) // 2, 2, -1])
        return self.loss_func(embeddings[:, 0], embeddings[:, 1], labels)

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view([len(samples) // 2, 2, -1])
        return 1 - torch.cosine_similarity(samples[:, 0], samples[:, 1])

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cosine_similarity(x[:, None, :], y[None, :, :], dim=2)

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self._ins_embedding.token_embedding.parameters())
        return {'total': total, 'token': token, 'model': total - token}

    @torch.no_grad()
    def update_queue(self, embeddings_y):
        start_index = self._embedding_queue_idx
        end_index = self._embedding_queue_idx + len(embeddings_y)
        new_index = torch.arange(start_index, end_index, device=embeddings_y.device) %\
                self._embedding_queue_size
        self._embedding_queue.index_copy_(0, new_index, embeddings_y)
        self._embedding_queue_idx = end_index % self._embedding_queue_size
        return new_index

    def info_nce_loss(self, embeddings:torch.Tensor, ids):
        distance = torch.cosine_similarity(embeddings[:, None, ], embeddings[None, :, :], dim=2)
        elements: torch.Tensor = torch.ones(size=(len(embeddings),),device=distance.device) * - torch.inf
        distance += torch.diag(elements)
        positive_index = torch.tensor([ i^1 for i in range(len(embeddings))], device=distance.device)
        loss = self.loss_func(distance / 0.1, positive_index)
        return loss

