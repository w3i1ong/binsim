import dgl
import torch
from torch import nn
from typing import Tuple, Any
from abc import ABC, abstractmethod

class GraphEmbeddingModelBase(nn.Module, ABC):
    def __init__(self, distance_func):
        super().__init__()
        self._distance_func = distance_func

    @classmethod
    def graphType(cls):
        raise

    @abstractmethod
    def sampleDataset(self):
        pass

    @property
    def distance_metric(self):
        return self._distance_func.metric

    def forward(self, samples: torch.Tensor):
        """
        Forward process of the model.
        :param samples: Batch of samples
        :return:
        """
        embeddings = self.generate_embedding(*samples)
        return embeddings

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

    def similarity(self, samples: torch.Tensor) -> torch.Tensor:
        samples = samples.view([len(samples) // 2, 2, samples.shape[-1]])
        return self._distance_func.similarity(samples[:, 0], samples[:, 1])

    def similarity_between_original(self, samples):
        sample_embeddings = self.generate_embedding(*samples)
        return self.similarity(sample_embeddings)

    def similarity_for_search(self, samples):
        samples = samples.view([len(samples) // 2, 2, samples.shape[-1]])
        return self._distance_func.similarity_for_search(samples[:, 0], samples[:, 1])

    def similarity_for_search_between_original(self, samples):
        sample_embeddings = self.generate_embedding(*samples)
        return self.similarity_for_search(sample_embeddings)

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._distance_func.pairwise_similarity(x, y)

    def pairwise_similarity_between_original(self, samples_x, samples_y):
        x_embeddings = self.generate_embedding(*samples_x)
        y_embeddings = self.generate_embedding(*samples_y)
        return self.pairwise_similarity(x_embeddings, y_embeddings)

    def pairwise_similarity_for_search(self, x, y):
        return self._distance_func.pairwise_similarity_for_search(x, y)

    def pairwise_similarity_for_search_between_original(self, samples_x, samples_y):
        x_embeddings = self.generate_embedding(*samples_x)
        y_embeddings = self.generate_embedding(*samples_y)
        return self.pairwise_similarity_for_search(x_embeddings, y_embeddings)

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
