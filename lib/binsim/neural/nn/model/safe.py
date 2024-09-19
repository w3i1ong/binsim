import torch
import torch.nn as nn
from typing import Union, Tuple
from binsim.neural.lm import Ins2vec
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from binsim.neural.nn.base.model import GraphEmbeddingModelBase


class SAFE(GraphEmbeddingModelBase):
    def __init__(self, ins2vec: Ins2vec,
                 distance_func,
                 out_dim: int = 100,
                 rnn_state_size: int = 50,
                 rnn_layers: int = 1,
                 attention_depth: int = 250,
                 attention_hops: int = 10,
                 dense_layer_size: int = 2000,
                 max_length=300,
                 need_att_weights=False,
                 device=None,
                 dtype=None):
        """
        An implementation of [SAFE](https://arxiv.org/abs/1811.05296).
        :param ins2vec: This argument provides the embedding matrix for the instructions. If a string is provided, it is
        assumed to be a path to a pickle file which contains the embedding matrix. Otherwise, it is assumed to be the
        embedding matrix.
        :param out_dim: The size of function embeddings.
        :param rnn_state_size: The size of the hidden state of the RNN.
        :param rnn_layers: The number of layers of the RNN.
        :param attention_depth: The depth of self-attention.
        :param attention_hops: The number of self-attention hops.
        :param dense_layer_size: The size of the dense layer.
        """
        super(SAFE, self).__init__(distance_func=distance_func)
        factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(ins2vec, str):
            ins2vec = Ins2vec.load(ins2vec)
        self.instructions_embeddings = ins2vec.as_torch_model(freeze=True, **factory_kwargs).to(device)
        ins2vec_dim = self.instructions_embeddings.embedding_dim

        self.bidirectional_rnn = torch.nn.GRU(
            input_size=ins2vec_dim,
            hidden_size=rnn_state_size,
            num_layers=rnn_layers,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
            **factory_kwargs
        )

        self.WS1 = nn.Linear(2 * rnn_state_size, attention_depth, **factory_kwargs, bias=False)
        self.WS2 = nn.Linear(attention_depth, attention_hops, **factory_kwargs, bias=False)
        self.attention_hops = attention_hops
        self.Wout1 = nn.Linear(
            2 * attention_hops * rnn_state_size,
            dense_layer_size,
            bias=False,
            **factory_kwargs
        )
        self.Wout2 = nn.Linear(
            dense_layer_size,
            out_dim,
            bias=False,
            **factory_kwargs
        )
        self._max_length = max_length
        self.need_att_weights = need_att_weights

    @property
    def graphType(self):
        from binsim.disassembly.utils.globals import GraphType
        return GraphType.TokenCFG

    @property
    def sampleDataset(self):
        from binsim.neural.utils.data import TokenSeqSampleDataset
        return TokenSeqSampleDataset

    @staticmethod
    def generate_mask(lengths: torch.Tensor, device=None, dtype=None):
        max_length = torch.max(lengths).item()
        range_matrix = torch.arange(0, max_length).expand(lengths.shape[0], max_length)
        return torch.lt(range_matrix, lengths.unsqueeze(1)).to(dtype=dtype, device=device)

    def generate_embedding(self, instructions: torch.Tensor,
                           lengths: torch.Tensor) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._max_length is not None and instructions.shape[1] > self._max_length:
            instructions = instructions[:, :self._max_length]
            lengths = torch.clip(lengths, max=self._max_length)
        lengths = lengths.cpu()

        # generate mask for attention sum
        mask = self.generate_mask(lengths, device=instructions.device, dtype=instructions.dtype)
        mask = torch.unsqueeze((1 - mask) * -1e6, dim=2)

        lengths, index = torch.sort(lengths, descending=True)
        _, old_index = torch.sort(index)
        instructions = instructions[index]

        instructions_vectors = self.instructions_embeddings(instructions)
        instructions_vectors = pack_padded_sequence(instructions_vectors, lengths, batch_first=True)
        output, _ = self.bidirectional_rnn(instructions_vectors)
        output = pad_packed_sequence(output, batch_first=True)
        output, _ = output
        output = output[old_index]
        output = output.reshape([output.shape[0], output.shape[1], -1, self.bidirectional_rnn.hidden_size * 2])[:, :,
                 -1]
        # shape: [batch, max_length, hop]
        weights = torch.softmax(self.WS2(torch.tanh(self.WS1(output))) + mask, dim=1)

        weights = torch.transpose(weights, 1, 2)  # shape: [batch, hop, max_length]

        temp = torch.reshape(torch.matmul(weights, output), (weights.size(0), -1))  # shape: [batch, hop*hidden_dim]
        function_embedding = self.Wout2(torch.relu(self.Wout1(temp)))
        if self.training and self.need_att_weights:
            return function_embedding, weights
        else:
            return function_embedding

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        token = sum(p.numel() for p in self.instructions_embeddings.parameters())
        return {'total': total, 'token': token, 'model': total - token}
