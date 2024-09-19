import torch
from .base import DistanceBase
from binsim.neural.nn.globals.siamese import EmbeddingDistanceMetric


class EuclidianDistance(DistanceBase):
    def __init__(self, eps=1e-8):
        super(EuclidianDistance, self).__init__()
        self._eps = eps

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, dim=-1, p=2)

    def similarity_for_search(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.similarity(x, y)

    def pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_pow = torch.pow(x, 2).sum(dim=-1)[..., None]
        y_pow = torch.pow(y, 2).sum(dim=-1)[None, ...]
        xy = torch.matmul(x, y.t())
        return torch.sqrt(torch.clip(x_pow + y_pow - 2 * xy, self._eps))

    def pairwise_similarity_for_search(self, x, y):
        return self.pairwise_similarity(x, y)

    @property
    def metric(self):
        return EmbeddingDistanceMetric.Euclidean

    def __str__(self):
        return 'Euclidean'
