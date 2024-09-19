import dgl, torch
from torch.nn import GRU
from dgl.ops import segment_reduce
from binsim.neural.nn.layer import DAGGRU, FastDAGGRU
from torch.nn.utils.rnn import pack_padded_sequence
from binsim.neural.nn.layer import InstructionEmbedding, Structure2vec
from binsim.neural.utils.data import BatchedInstruction, BatchedBBIndex
from binsim.neural.nn.base.model import GraphEmbeddingModelBase


class RCFG2Vec(GraphEmbeddingModelBase):
    def __init__(self, in_dim,
                 embed_dim,
                 vocab_size,
                 distance_func,
                 layer_num=1,
                 rnn_layers=2,
                 gru_bidirectional=True,
                 dag_gru_bidirectional=False,
                 message_passing=True,
                 divide_in_degree=True,
                 use_dag=True,
                 use_fast=False,
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
        super(RCFG2Vec, self).__init__(distance_func=distance_func)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # initialize the instruction embedding layer
        self._ins_embedding = InstructionEmbedding(vocab_size, in_dim, **factory_kwargs)
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
        self._use_fast = use_fast
        if use_dag:
            if use_fast:
                self.dag_gru = FastDAGGRU(embed_dim, embed_dim, layer_num=layer_num, bidirectional=dag_gru_bidirectional,
                                          divide_in_degree=divide_in_degree, message_passing=message_passing,
                                          **factory_kwargs)
            else:
                self.dag_gru = DAGGRU(embed_dim, embed_dim, layer_num=layer_num, bidirectional=dag_gru_bidirectional,
                                      divide_in_degree=divide_in_degree, message_passing=message_passing,
                                      **factory_kwargs)
        else:
            self.dag_gru = Structure2vec(embed_dim, embed_dim, iteration_round=layer_num, **factory_kwargs)

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.InsCFG

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data.dataset import InsCFGSampleDataset
        return InsCFGSampleDataset

    def generate_basic_block_embeddings(self, unique_ins_embedding: torch.Tensor,
                                        basic_block_ins_index: BatchedBBIndex):
        basic_block_embedding_chunks = []
        for bb_index_chunk, bb_length_chunk in basic_block_ins_index.bb_chunks:
            bb_length_chunk = bb_length_chunk.cpu().to(torch.int64)
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

        if not self._use_fast:
            if getattr(self, '_use_dag', True):
                assert 'node_id' in graph.ndata, "When using DAGGRU, the graph must have nodeId features."
                features = features[graph.ndata['node_id']]
            # generate graph embedding
            features = self.dag_gru(graph, features)
            graph.ndata['h'] = features
            embeddings = dgl.sum_nodes(graph, 'h')
            return embeddings
        else:
            features = features[graph.node_ids]
            features = self.dag_gru(graph, features)
            node_num = torch.zeros((graph.graph_ids.max()+1,), dtype=torch.long)
            ones = torch.ones_like(graph.graph_ids).to(torch.long)
            node_num = torch.index_add(node_num, 0, graph.graph_ids, ones).cuda()
            embeddings = segment_reduce(node_num, features, "mean")
            return embeddings


    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self._ins_embedding.token_embedding.parameters())
        return {'total': total, 'token': token, 'model': total - token}
