import torch
from .base import DistanceBase
from torch import Tensor
from binsim.neural.nn.globals.siamese import SiameseSampleFormat, EmbeddingDistanceMetric


class CosineDistance(DistanceBase):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self._eps = eps

    def similarity(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cosine_similarity(x, y, dim=-1, eps=self._eps)

    def similarity_for_search(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.similarity(x, y)

    def pairwise_similarity(self, x: Tensor, y: Tensor) -> Tensor:
        return x @ y.t() / (x.norm(dim=-1)[:, None] * y.norm(dim=-1)[None] + self._eps)

    def pairwise_similarity_for_search(self, x, y):
        return 1 - self.pairwise_similarity(x, y)

    @property
    def metric(self):
        return EmbeddingDistanceMetric.Cosine

    def __str__(self):
        return 'Cosine'
