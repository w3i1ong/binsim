import torch
from torch.nn import CosineSimilarity as CosineDistance
from torch import Tensor


class PairwiseCosineDistance(CosineDistance):
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super(PairwiseCosineDistance, self).__init__(dim=1)
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = x1 / torch.norm(x1, p=2, dim=self.dim, keepdim=True)
        x2 = x2 / torch.norm(x2, p=2, dim=self.dim, keepdim=True)
        return x1 @ x2.T
