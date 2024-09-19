import torch
from torch import nn
from .base import ContrastiveLossBase
from binsim.neural.nn.globals.siamese import SiameseSampleFormat, EmbeddingDistanceMetric

class SAFELoss(ContrastiveLossBase):
    def __init__(self):
        super().__init__()
        self._loss = nn.MSELoss(reduction='sum')

    def check(self, sample_format, distance_metric):
        assert sample_format == SiameseSampleFormat.Pair, "When using MSELoss, the sample format must be Pair."
        assert distance_metric == EmbeddingDistanceMetric.Cosine, \
            f"Distance metric {distance_metric} is not supported by MSELoss."

    def __call__(self, samples, labels, sample_ids=None, pair_sim_func=None, pairwise_sim_func=None):
        assert isinstance(samples, tuple)
        embeddings, attention_weights = samples
        distance = pair_sim_func(embeddings)

        penalty = torch.bmm(attention_weights, attention_weights.permute(0,2,1))
        head_num = attention_weights.shape[1]
        penalty -= torch.eye(head_num, device=penalty.device).unsqueeze(0)

        return self._loss(distance, labels * 2 - 1) + torch.norm(penalty, p=2, dim=(1,2)).sum()


class AsteriaLoss(ContrastiveLossBase):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCELoss(reduction="mean")

    def check(self, sample_format, distance_metric):
        assert sample_format == SiameseSampleFormat.Pair, "When using MSELoss, the sample format must be Pair."
        assert distance_metric == EmbeddingDistanceMetric.AsteriaDistance, \
            f"Distance metric {distance_metric} is not supported by MSELoss."

    def __call__(self, samples, labels, sample_ids=None, pair_sim_func=None, pairwise_sim_func=None):
        labels = labels.float()
        similarity = pair_sim_func(samples)
        probability = torch.stack([1-similarity, similarity], dim=1)
        target = torch.stack([1-labels, labels], dim=1)
        return self._loss(probability, target)
