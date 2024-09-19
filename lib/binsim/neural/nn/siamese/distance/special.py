import torch
from torch import nn
from .base import DistanceBase
from binsim.neural.nn.globals.siamese import EmbeddingDistanceMetric

class AsteriaDistance(DistanceBase):
    def __init__(self, embed_size):
        super(AsteriaDistance, self).__init__()
        self.W = nn.Linear(embed_size * 2,2)

    def real_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, callee_num_x = x[:, :-1], x[:, -1]
        y, callee_num_y = y[:, :-1], y[:, -1]
        logit = self.W(torch.sigmoid(torch.cat([(x-y).abs(), x*y], dim=-1)))
        confident_score = torch.softmax(logit, dim=1)[:,1]
        return confident_score

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.real_similarity(x, y)

    def similarity_for_search(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - self.real_similarity(x, y)

    def real_pairwise_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = x[:, None], y[None]
        x, callee_num_x = x[:,:,:-1], x[:,:,-1]
        y, callee_num_y = y[:,:,:-1], y[:,:,-1]
        logit = self.W(torch.sigmoid(torch.cat([(x-y).abs(), x*y], dim=-1)))
        confident_score = torch.softmax(logit, dim=1)[:,:, 1]
        return confident_score

    def pairwise_similarity(self, x, y):
        return self.real_pairwise_similarity(x,y)

    def pairwise_similarity_for_search(self, x, y):
        return 1 - self.real_pairwise_similarity(x, y)

    @property
    def metric(self):
        return EmbeddingDistanceMetric.AsteriaDistance

    def __str__(self):
        return 'AsteriaDistance'
