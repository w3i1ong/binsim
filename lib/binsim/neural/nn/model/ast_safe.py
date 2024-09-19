import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from binsim.neural.nn.layer import InstructionEmbedding
from binsim.neural.utils.data import BatchedInstruction

class ASTSAFE(GraphEmbeddingModelBase):
    def __init__(self, in_dim,
                 vocab_size,
                 distance_func,
                 out_dim: int = 100,
                 rnn_layers: int = 1,
                 device=None,
                 dtype=None):
        """
        An implementation of [SAFE](https://arxiv.org/abs/1811.05296).
        :param out_dim: The size of function embeddings.
        :param rnn_layers: The number of layers of the RNN.
        """
        super(ASTSAFE, self).__init__(distance_func=distance_func)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self._ins_embedding = InstructionEmbedding(vocab_size, in_dim, **factory_kwargs)

        self.gru = torch.nn.GRU(
            input_size=in_dim,
            hidden_size=out_dim // 2,
            num_layers=rnn_layers,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
            **factory_kwargs
        )
        self.loss_func = nn.MSELoss(reduction='mean')

    @property
    def graphType(self):
        raise NotImplementedError(f"graphType has not been implemented for {self.__class__}")

    @property
    def sampleDataset(self):
        raise NotImplementedError(f"sampleDataloader has not been implemented for {self.__class__}")

    @staticmethod
    def generate_mask(lengths: torch.Tensor, device=None, dtype=None):
        max_length = torch.max(lengths).item()
        range_matrix = torch.arange(0, max_length, device=lengths.device).expand(lengths.shape[0], max_length)
        return torch.lt(range_matrix, lengths.unsqueeze(1))

    def generate_basic_block_embeddings(self, unique_ins_embedding: torch.Tensor,
                                        basic_block_ins_index):
        basic_block_embedding_chunks = []
        for bb_index_chunk, bb_length_chunk in basic_block_ins_index.bb_chunks:
            embedding_seq = unique_ins_embedding[bb_index_chunk]
            embedding_seq = pack_padded_sequence(embedding_seq, bb_length_chunk.cpu(), batch_first=True)
            _, features = self.gru(embedding_seq)
            features = torch.transpose(features, 0, 1)
            basic_block_embedding_chunks.append(features)
        basic_block_embeddings = torch.cat(basic_block_embedding_chunks, dim=0)
        return basic_block_embeddings[basic_block_ins_index.original_index]

    def generate_embedding(self, batched_instructions: BatchedInstruction,
                           basic_block_ins_index) -> torch.Tensor:
        unique_instruction_embeddings = self._ins_embedding(batched_instructions)
        function_embeddings = self.generate_basic_block_embeddings(unique_instruction_embeddings, basic_block_ins_index)
        function_embeddings = function_embeddings[:, -2:, :]
        function_embeddings = function_embeddings.reshape(function_embeddings.shape[0], -1)
        return function_embeddings

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
