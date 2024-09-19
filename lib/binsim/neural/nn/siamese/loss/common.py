import torch
from torch import nn
from .base import ContrastiveLossBase
from binsim.neural.nn.globals.siamese import SiameseSampleFormat, EmbeddingDistanceMetric

class MSELoss(ContrastiveLossBase):
    def __init__(self):
        super().__init__()
        self._loss = nn.MSELoss()

    def check(self, sample_format, distance_metric):
        assert sample_format == SiameseSampleFormat.Pair, "When using MSELoss, the sample format must be Pair."
        assert distance_metric == EmbeddingDistanceMetric.Cosine, \
            f"Distance metric {distance_metric} is not supported by MSELoss."

    def __call__(self, samples, labels, sample_ids=None, pair_sim_func=None, pairwise_sim_func=None):
        distance = pair_sim_func(samples)
        return self._loss(distance, labels * 2 - 1)


class CosineEmbeddingLoss(ContrastiveLossBase):
    def __init__(self, margin: float):
        super().__init__()
        self._margin = margin
        self._loss = nn.CosineEmbeddingLoss(margin=margin)

    def check(self, sample_format, distance_metric):
        assert sample_format == SiameseSampleFormat.Pair, "When using CosineEmbeddingLoss, the sample format must be Pair."
        assert distance_metric == EmbeddingDistanceMetric.Cosine, \
            f"Distance metric {distance_metric} is not supported by CosineEmbeddingLoss."

    def __call__(self, samples, labels, sample_ids=None, pair_sim_func=None, pairwise_sim_func=None):
        samples = samples.view(-1, 2, samples.shape[-1])
        x, y = samples[:, 0], samples[:, 1]
        labels = labels * 2 - 1
        return self._loss(x, y, target=labels)

class InfoNCELoss(ContrastiveLossBase):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self._loss = nn.CrossEntropyLoss()
        self._temperature = temperature

    def check(self, sample_format, distance_metric):
        assert sample_format in [SiameseSampleFormat.QueryTarget, SiameseSampleFormat.PositivePair], \
            "When using InfoNCELoss, the sample format must be QueryTarget or PositivePair, but got {sample_format}."
        assert distance_metric == EmbeddingDistanceMetric.Cosine, \
            f"InfoNCELoss only supports cosine distance metric, but got {distance_metric}."

    def __call__(self, samples, match_pos=None, sample_ids=None, labels=None, pair_sim_func=None, pairwise_sim_func=None):
        if isinstance(samples, tuple):
            x, y = samples
            pairwise_sim = pairwise_sim_func(x, y) / self._temperature
            return self._loss(pairwise_sim, match_pos)
        else:
            mask = sample_ids[:, None] == sample_ids[None, :]
            positive_index = torch.arange(0, len(sample_ids), device=sample_ids.device)
            kept_mask = positive_index[:, None] == (positive_index[None, :] ^ 1)
            mask = torch.bitwise_and(mask, torch.bitwise_not(kept_mask)).to(samples.device)
            distance = pairwise_sim_func(samples, samples) / self._temperature
            distance = torch.where(mask, torch.tensor(float('-inf'),device=distance.device), distance)
            loss = self._loss(distance , positive_index.to(samples.device) ^ 1)
            return loss

class TripletLoss(ContrastiveLossBase):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self._margin = margin
        self._loss = nn.TripletMarginWithDistanceLoss(margin=margin, distance_function=nn.PairwiseDistance(p=2))

    def check(self, sample_format, distance_metric):
        assert sample_format == SiameseSampleFormat.Triplet, "When using TripletLoss, the sample format must be Triplet."
        assert distance_metric == EmbeddingDistanceMetric.Euclidean, \
            f"TripletLoss only supports cosine distance metric, but got {distance_metric}."

    def __call__(self, samples, labels, sample_ids=None, pair_sim_func=None, pairwise_sim_func=None):
        samples = samples.view(-1, 3, samples.shape[-1])
        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        return self._loss(x, y, z)
