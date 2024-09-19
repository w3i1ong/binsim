import torch
from torch.nn import Module


class PairwiseEuclidianDistance(Module):
    def __init__(self):
        super(PairwiseEuclidianDistance, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_2 = torch.sum(x ** 2, dim=1, keepdim=True)
        y_2 = torch.sum(y ** 2, dim=1, keepdim=True).T
        xy = x @ y.T
        return torch.sqrt(torch.clip(x_2 + y_2 - 2 * xy, 1e-6))


class EuclidianDistance(Module):
    def __init__(self):
        super(EuclidianDistance, self).__init__()

    def forward(self, x, y):
        return torch.sqrt(torch.clip(torch.sum((x - y) ** 2, dim=1), 1e-6))
