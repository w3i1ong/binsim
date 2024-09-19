import torch
import torch.nn as nn
from typing import List
from binsim.neural.nn.base.model import GraphEmbeddingModelBase


class AlphaDiff(GraphEmbeddingModelBase):
    def __init__(self, out_dim, distance_func, dtype=None, device=None, epsilon=0.75):
        """
        :param out_dim: The dimension of the output embedding.
        :param distance_func: The distance function used to calculate the similarity between two embeddings.
        :param dtype: The data type of the model parameters.
        :param device: The device of the model parameters.
        :param epsilon: The epsilon value used in the similarity calculation.
        An implementation of [Alpha-diff](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9000005).
        """
        super(AlphaDiff, self).__init__(distance_func=distance_func)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # The structure of this model can be found at https://twelveand0.github.io/AlphaDiff-ASE2018-Appendix.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, **factory_kwargs),
            nn.BatchNorm2d(32, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, **factory_kwargs),
            nn.BatchNorm2d(32, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, **factory_kwargs),
            nn.BatchNorm2d(64, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, **factory_kwargs),
            nn.BatchNorm2d(64, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 96, 3, stride=1, padding=1, **factory_kwargs),
            nn.BatchNorm2d(96, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1, **factory_kwargs),
            nn.BatchNorm2d(96, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(96, 96, 3, stride=1, padding=1, **factory_kwargs),
            nn.BatchNorm2d(96, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1, **factory_kwargs),
            nn.BatchNorm2d(96, **factory_kwargs),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.flatten_layers = nn.Sequential(
            nn.Linear(96, 512, bias=True, **factory_kwargs),
            nn.Flatten(),
            nn.Linear(18432, out_dim, bias=True, **factory_kwargs)
        )
        self._epsilon = epsilon


    @property
    def graphType(self):
        raise NotImplementedError(f"graphType has not been implemented for {self.__class__}")

    @property
    def sampleDataset(self):
        raise NotImplementedError(f"sampleDataloader has not been implemented for {self.__class__}")

    def generate_embedding(self, bytecode: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        """
        Calculate the embedding for given bytecode.
        :param bytecode: A Tensor of [batch_size, 10000]. The bytecode sequences of functions to be processed.
        :param degrees: A Tensor of [batch_size, 2]. The caller-number and callee-number of functions.
        :return:
        """
        bytecode = (bytecode - 256) / 128
        assert bytecode.shape[1] == 10000
        bytecode = torch.reshape(bytecode, [bytecode.shape[0], 1, 100, 100])
        embedding = self.conv_layers(bytecode)
        embedding = torch.transpose(embedding, 1, 3)
        return torch.cat([self.flatten_layers(embedding), degrees], dim=1)

    def similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate the similarity between a batch of sample pairs.
        :param embeddings: A batch of embeddings, with shape [batch_size*2, embedding_size].
        :return: Similarity between each sample pair, with shape [batch_size].
        """
        embeddings = embeddings[:, :-2]
        return super().similarity(embeddings)

    def similarity_for_search(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings, degrees = embeddings[:, :-2], embeddings[:, -2:]
        x_degrees, y_degrees = degrees[::2], degrees[1::2]
        return (super().similarity_for_search(embeddings) +
                1 - self._epsilon ** torch.sqrt(torch.sum(x_degrees[:, None, :] - y_degrees[None, :], dim=-1) ** 2))

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x[:, :-2], y[:, :-2]
        return super().pairwise_similarity(x, y)

    def pairwise_similarity_for_search(self, x: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the pairwise similarity between two batch of samples.
        :param x: A batch of embeddings, with shape [batch_size_x, embedding_size].
        :param y: A batch of embeddings, with shape [batch_size_y, embedding_size].
        :return: Pairwise similarity between each sample pair, with shape [batch_size_x, batch_size_y].
        """
        x, x_degrees = x[:, :-2], x[:, -2:]
        y, y_degrees = y[:, :-2], y[:, -2:]
        return (super().pairwise_similarity(x, y) +
                1 - self._epsilon ** torch.sqrt(torch.sum(x_degrees[:, None, :] - y_degrees[None, :], dim=-1) ** 2))

    @property
    def parameter_statistics(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total}
