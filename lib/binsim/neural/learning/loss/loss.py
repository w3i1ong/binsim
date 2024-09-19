import torch
from torch.nn import MSELoss, CosineSimilarity, PairwiseDistance, CosineEmbeddingLoss
from torch.nn import Module

class CosineMSELoss(Module):
    def __init__(self):
        super(CosineMSELoss, self).__init__()
        self._loss = MSELoss()

    def forward(self, anchor, another, labels):
        distance = torch.cosine_similarity(anchor, another)
        return self._loss(distance, labels)
