import torch
from torch import Tensor
from .base import SamplerBase
from typing import Callable


class OnlineSemiHardNegativeSampler(SamplerBase):
    def __init__(self, margin=0.5):
        super().__init__()
        self._margin = margin

    def __call__(self, anchors: Tensor,
                 anchor_labels: Tensor,
                 positive: Tensor,
                 positive_labels: Tensor,
                 pair_sim_func: Callable
                 ) -> Tensor:
        """
        An implementation of Online semi hard sampling.
        :param anchors: $n$ embeddings for anchor samples
        :param anchor_labels: $n$ labels for anchor samples.
        :param positive: $n$ embeddings for positive samples. The $i$-th positive sample is in the same class with the
            $i$-th anchor sample.
        :param positive_labels: The labels for positive samples. This should be same with anchor_labels.
        :param pair_sim_func: Callable function that can be used to calculate the pairwise similarity score between
            anchor embeddings and positive embeddings.
        :return:
        """
        margin = torch.tensor(self._margin, device=anchors.device)
        anchor_labels, positive_labels = anchor_labels.to(anchors.device), positive_labels.to(anchors.device)
        with torch.no_grad():
            # compute similarities
            pair_sim = pair_sim_func(anchors, positive)
            # shape: [batch, batch]
            sample_diff = torch.not_equal(anchor_labels.unsqueeze(1), positive_labels.unsqueeze(0))
            # shape: [batch, batch], diff[i][j] = 1 if anchor_labels[i] != positive_labels[j]
            anchor_positive_sim = torch.diag(pair_sim).unsqueeze(1)
            # shape: [batch, 1], anchor_positive_sim[i] = pair_sim[i][i]
            # easy pairs
            easy_pairs = torch.bitwise_and(torch.gt(pair_sim, margin), sample_diff)
            # shape: [batch, batch], easy_pairs[i][j] = 1 if pair_sim[i][j] > margin and anchor_labels[i] != positive_labels[j]
            # hard pairs
            hard_pairs = torch.bitwise_and(torch.lt(pair_sim, anchor_positive_sim), sample_diff)
            # shape: [batch, batch], hard_pairs[i][j] = 1 if pair_sim[i][j] < anchor_positive_sim[i] and anchor_labels[i] != positive_labels[j]
            # semi-hard pairs
            semi_hard_pairs = torch.bitwise_and(torch.le(pair_sim, margin), torch.ge(pair_sim, anchor_positive_sim))
            semi_hard_pairs = torch.bitwise_and(semi_hard_pairs, sample_diff)
            # shape: [batch, batch], semi_hard_pairs[i][j] = 1 if margin >= pair_sim[i][j] >= anchor_positive_sim[i] and anchor_labels[i] != positive_labels[j]

            semi_rows = torch.sum(semi_hard_pairs, dim=1).bool()
            # shape: [batch], semi_rows[i] = 1 if there is at least one semi-hard pair in the i-th row
            hard_rows = torch.sum(hard_pairs, dim=1).bool()
            # shape: [batch], hard_rows[i] = 1 if there is at least one hard pair in the i-th row
            # easy_rows = torch.sum(easy_pairs, dim=1).bool()
            # shape: [batch], easy_rows[i] = 1 if there is at least one easy pair in the i-th row

            min_sim, _ = torch.min(pair_sim, 1, keepdim=True)
            max_sim, _ = torch.max(pair_sim, 1, keepdim=True)

            _, semi_hard_max_index = \
                torch.max((pair_sim - min_sim) * semi_hard_pairs.float(), 1)
            # shape: [batch, 1], semi_hard_min_distance[i] = min(pair_sim[i][j] - max_sim[i] for j in range(batch) if semi_hard_pairs[i][j] == 1)

            _, easy_min_idx = torch.min((pair_sim - max_sim) * easy_pairs.float(), 1)
            # shape: [batch, 1], easy_min_distance[i] = min(pair_sim[i][j] - max_sim[i] for j in range(batch) if easy_pairs[i][j] == 1)

            _, hard_max_index = torch.max((pair_sim - min_sim) * hard_pairs.float(), 1)

            negative_idx = torch.where(hard_rows, hard_max_index, easy_min_idx)
            negative_idx = torch.where(semi_rows, semi_hard_max_index, negative_idx)
        return negative_idx
