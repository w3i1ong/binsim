import dgl
import torch
from typing import Tuple, Any
from torch import nn
from abc import ABC, abstractmethod
from binsim.neural.nn.globals.siamese import SiameseSampleFormat
from binsim.neural.learning.sampling import OnlineSemiHardNegativeSampler


class GraphEmbeddingModelBase(nn.Module, ABC):
    def __init__(self, sample_format: SiameseSampleFormat):
        """
        :param sample_format: If provided, the model will be trained using triplet loss. Otherwise, siamese loss will be used.
        """
        super().__init__()
        self.sample_format = sample_format
        self._margin = None

    @classmethod
    def graphType(cls):
        raise

    @abstractmethod
    def pairDataset(self):
        pass

    @abstractmethod
    def sampleDataset(self):
        pass

    @property
    def margin(self):
        if self._margin is None:
            raise ValueError("The margin should be set before used.")
        return self._margin

    @margin.setter
    def margin(self, value):
        self._margin = value

    def _semi_hard_sampler(self, anchors, positives, anchor_ids, positive_ids: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampler = OnlineSemiHardNegativeSampler(margin=self.margin)
        negative_idx = sampler(anchors, anchor_ids, positives, positive_ids, self.pairwise_similarity).cpu()
        return (torch.arange(0, len(anchors)),
                torch.arange(0, len(anchors)), negative_idx)

    def _semi_hard_triplet_sampler(self, embeddings: torch.Tensor, labels: torch.Tensor, ids: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def sampler(self):
        match self.sample_format:
            case SiameseSampleFormat.SemiHardPair:
                return self._semi_hard_sampler
            case SiameseSampleFormat.SemiHardTriplet:
                return self._semi_hard_triplet_sampler
            case _:
                raise NotImplementedError

    @property
    def sample_format(self):
        return self._sample_format

    @sample_format.setter
    def sample_format(self, value):
        if value is None:
            value = SiameseSampleFormat.Pair

        assert isinstance(value, (SiameseSampleFormat, str)), \
            "sample_format must be an instance of SiameseSampleFormat or a string."

        if isinstance(value, str):
            value = SiameseSampleFormat(value)
        self._sample_format = value

    def forward(self, samples: torch.Tensor, labels: torch.Tensor, ids: torch.Tensor):
        """
        Forward process of the model.
        :param samples: Batch of samples
        :param labels:
        :param ids:
        :return:
        """
        embeddings = self.generate_embedding(*samples)

        extra_data = []
        if isinstance(embeddings, tuple):
            embeddings, *extra_data = embeddings

        match self._sample_format:
            case SiameseSampleFormat.Pair:
                return self.siamese_loss(embeddings, labels, ids, *extra_data)
            case SiameseSampleFormat.Triplet:
                return self.triplet_loss(embeddings, labels, ids, *extra_data)
            case SiameseSampleFormat.SemiHardPair:
                sampled_embeddings, sampled_ids, labels = self.generate_semi_hard_tuples(self.sampler, embeddings, ids)
                return self.siamese_loss(sampled_embeddings, labels, sampled_ids, *extra_data)
            case SiameseSampleFormat.SemiHardTriplet:
                sampled_embeddings, sampled_ids = self.generate_semi_hard_tuples(self.sampler, embeddings, ids)
                return self.triplet_loss(sampled_embeddings, labels, sampled_ids, *extra_data)
            case SiameseSampleFormat.InfoNCESamples:
                return self.info_nce_loss(embeddings, ids)

    @staticmethod
    def generate_semi_hard_tuples(sampler, embeddings, ids, triplet=False):
        embedding_size = embeddings.shape[-1]
        embeddings = embeddings.reshape([len(embeddings) // 2, 2, -1])
        ids = ids.reshape([len(embeddings), 2])
        ids_x, ids_y = ids[:, 0], ids[:, 1]
        embeddings_x, embeddings_y = embeddings[:, 0], embeddings[:, 1]

        anchor_idx, positive_index, negative_index = \
                sampler(embeddings_x, embeddings_y, ids_x, ids_y)
        anchor_ids, negative_ids, positive_ids = ids_x[anchor_idx], ids_y[negative_index], ids_y[positive_index]
        anchor_embeddings, negative_embeddings, positive_embeddings = (embeddings_x[anchor_idx],
                                                                       embeddings_y[positive_index], embeddings_y[negative_index])

        if triplet:
            ids = torch.stack([anchor_embeddings, positive_embeddings, negative_embeddings], dim=1)
            embeddings = torch.stack([anchor_ids, positive_ids, negative_ids], dim=1)
        else:
            ids = torch.stack([anchor_ids, positive_ids, anchor_ids, negative_ids], dim=1)
            embeddings = torch.stack([anchor_embeddings, positive_embeddings, anchor_embeddings, negative_embeddings], dim=1)
        ids = torch.reshape(ids,[-1])
        embeddings = torch.reshape(embeddings, [-1, embedding_size])

        if triplet:
            return embeddings, ids
        else:
            ones = torch.ones([len(embeddings) // 4, 1], device=embeddings.device)
            zeros = torch.zeros([len(embeddings) // 4, 1], device=embeddings.device)
            labels = torch.cat([zeros, ones], dim=1).reshape([-1])
            return embeddings, ids, labels

    @abstractmethod
    def generate_embedding(self, *args) -> torch.Tensor:
        pass

    @abstractmethod
    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        pass

    def similarity_between_original(self, samples):
        sample_embeddings = self.generate_embedding(*samples)
        return self.similarity(sample_embeddings)

    @abstractmethod
    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def pairwise_similarity_between_original(self, samples_x, samples_y):
        x_embeddings = self.generate_embedding(*samples_x)
        y_embeddings = self.generate_embedding(*samples_y)
        return self.pairwise_similarity(x_embeddings, y_embeddings)

    def siamese_loss(self, samples: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Siamese loss is not implemented for {self.__class__}!")

    def triplet_loss(self, anchors: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Triplet loss is not implemented for {self.__class__}!")

    @property
    def parameter_statistics(self):
        raise NotImplementedError("The method parameter_statistics is not implemented for GraphMatchingModelBase!")


    def info_nce_loss(self, embeddings, ids):
        raise NotImplementedError(f"inoNCE loss has not been implemented for {self.__class__}!")


    @staticmethod
    def from_pretrained(filename: str, device=None):
        return torch.load(filename, map_location=device)

    def save(self, filename: str):
        torch.save(self, filename)

class GraphMatchingModelBase(nn.Module, ABC):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._out_dim = out_dim

    def forward(self, samples: Tuple[dgl.DGLGraph, Any, Any], labels, sample_ids=None):
        graph, node_features, edge_features = samples
        match_embeddings = self._generate_match_embedding(graph, node_features, edge_features)
        return self.siamese_loss(match_embeddings, labels, sample_ids)

    @abstractmethod
    def _generate_match_embedding(self, graph: dgl.DGLGraph, node_features: Any, edge_features: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def siamese_loss(self, samples, labels: torch.Tensor, sample_ids) -> torch.Tensor:
        pass

    def triplet_loss(self, samples, sample_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Triplet loss is not implemented for GraphMatchingNet!")

    @abstractmethod
    def similarity(self, samples) -> torch.Tensor:
        """
        Calculate the similarity between a batch of sample pairs.
        :param graph: A batched graph. The 2k-th graph and the 2k+1-th graph consist of a sample pair.
        :param node_features: Tensor of node features, shape [total_num_nodes, node_feature_size].
        :param edge_features: Tensor of edge features, shape [total_num_edges, edge_feature_size].
        :return: Similarity between each sample pair, shape [batch_size//2].
        """
        pass

    @abstractmethod
    def pairwise_similarity(self, x_samples: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor],
                            y_samples: Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the pairwise similarity between two batch of samples.
        :param x_samples: A tuple of (graph, node_features, edge_features), where graph is a batched graph,
            node_features is a tensor of node features, and edge_features is a tensor of edge features.
        :param y_samples: A tuple of (graph, node_features, edge_features), where graph is a batched graph,
            node_features is a tensor of node features, and edge_features is a tensor of edge features.
        :return: Pairwise similarity between each sample pair, shape [num_samples_x, num_samples_y].
        """
        pass

    def pairwise_similarity_between_original(self, samples_x, samples_y):
        return self.pairwise_similarity(samples_x, samples_y)
