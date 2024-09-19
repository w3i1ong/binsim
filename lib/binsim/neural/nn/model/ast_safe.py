import torch
import torch.nn as nn
from typing import Union, Tuple
from binsim.neural.lm import Ins2vec
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from binsim.neural.nn.base.model import GraphEmbeddingModelBase
from binsim.neural.nn.layer import InstructionEmbedding, Structure2vec
from binsim.neural.utils.data import BatchedInstruction, BatchedBBIndex

class ASTSAFE(GraphEmbeddingModelBase):
    def __init__(self, in_dim,
                 vocab_size,
                 out_dim: int = 100,
                 rnn_state_size: int = 50,
                 rnn_layers: int = 1,
                 attention_depth: int = 250,
                 attention_hops: int = 10,
                 dense_layer_size: int = 2000,
                 sample_format=None,
                 max_length=250,
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
        super(ASTSAFE, self).__init__(sample_format=sample_format)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self._ins_embedding = InstructionEmbedding(vocab_size, in_dim, **factory_kwargs)

        self.bidirectional_rnn = torch.nn.GRU(
            input_size=in_dim,
            hidden_size=rnn_state_size,
            num_layers=rnn_layers,
            bias=True,
            batch_first=True,
            dropout=0,
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
        self.loss_func = nn.MSELoss(reduction='mean')
        self._max_length = max_length
        self.margin = 0.5

    @property
    def graphType(self):
        raise NotImplementedError(f"graphType has not been implemented for {self.__class__}")

    @property
    def pairDataset(self):
        raise NotImplementedError(f"pairDataloader has not been implemented for {self.__class__}")

    @property
    def sampleDataset(self):
        raise NotImplementedError(f"sampleDataloader has not been implemented for {self.__class__}")

    @staticmethod
    def generate_mask(lengths: torch.Tensor, device=None, dtype=None):
        max_length = torch.max(lengths).item()
        range_matrix = torch.arange(0, max_length, device=lengths.device).expand(lengths.shape[0], max_length)
        return torch.lt(range_matrix, lengths.unsqueeze(1))

    def generate_embedding(self, batched_instructions: BatchedInstruction,
                           basic_block_ins_index: torch.Tensor,
                           lengths: torch.Tensor) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lengths = lengths.cpu()
        unique_instruction_embeddings = self._ins_embedding(batched_instructions)
        instructions = unique_instruction_embeddings[basic_block_ins_index]
        # generate mask for attention sum
        mask = self.generate_mask(lengths).float().to(instructions.device)
        mask = torch.unsqueeze((1 - mask) * -1e6, dim=2)

        lengths, index = torch.sort(lengths, descending=True)
        _, old_index = torch.sort(index)
        instructions_vectors = instructions[index]

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
        if self.training:
            return function_embedding, weights
        else:
            return function_embedding

    def siamese_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor,
                     *extra_args) -> torch.Tensor:
        # weights = extra_args[0]
        labels = torch.mul(torch.sub(labels, 0.5), 2)
        # A_A_T = torch.bmm(weights, torch.permute(weights, (0, 2, 1))) - torch.unsqueeze(torch.eye(self.attention_hops, device=labels.device), dim=0)
        # the original paper uses the following loss, but it performs bad in our experiments, so we remove the penalty term.
        # return self.loss_func(labels, 1 - self.similarity(embeddings)) + torch.mean(torch.norm(A_A_T, 2, dim=(1,2)))
        embeddings = torch.reshape(embeddings, [embeddings.shape[0] // 2, 2, -1])
        sample1, sample2 = embeddings[:, 0], embeddings[:, 1]
        return self.loss_func(torch.cosine_similarity(sample1, sample2), labels)

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view([len(samples) // 2, 2, -1])
        return 1 - torch.cosine_similarity(samples[:, 0], samples[:, 1])

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x / torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + torch.tensor(1e-08, device=x.device))
        y = y / torch.sqrt(torch.sum(y ** 2, dim=1, keepdim=True) + torch.tensor(1e-08, device=x.device))
        return 1 - x @ y.T

    def triplet_loss(self, anchors: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
