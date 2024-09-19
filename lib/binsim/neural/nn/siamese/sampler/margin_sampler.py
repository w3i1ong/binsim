import torch
from .base import SamplerBase
from binsim.neural.nn.globals.siamese import SiameseSampleFormat, EmbeddingDistanceMetric

class MarginSampler(SamplerBase):
    def __init__(self, margin: float, triple=False):
        self._margin = margin
        self._triple = triple

    def __call__(self, embeddings, sample_ids, labels, pair_sim_func, sample_format, distance_metric):
        sample_num, embedding_dim = embeddings.size()
        assert sample_format == SiameseSampleFormat.PositivePair, f"MarginSampler only supports pair samples, but got {sample_format}."
        assert embeddings.size(0) % 2 == 0, f"Expected even number of embeddings, but got {embeddings.size(0)}."
        embeddings = embeddings.resize(sample_num // 2, 2, embedding_dim)
        sample_ids = sample_ids.resize(sample_num // 2, 2)
        anchors, positive = embeddings[:, 0], embeddings[:, 1]
        anchor_ids, positive_ids = sample_ids[:, 0], sample_ids[:, 1]


        with torch.no_grad():
            margin = torch.tensor(self._margin, device=embeddings.device)
            # compute similarities
            pair_sim = pair_sim_func(anchors, positive)
            match distance_metric:
                case EmbeddingDistanceMetric.Cosine:
                    pair_sim = 1 - pair_sim
                case EmbeddingDistanceMetric.Euclidean:
                    pair_sim = pair_sim
                case EmbeddingDistanceMetric.SelfDefined:
                    raise NotImplementedError
                case _:
                    raise ValueError
            # shape: [batch, batch]
            sample_diff = torch.not_equal(anchor_ids.unsqueeze(1), positive_ids.unsqueeze(0)).to(pair_sim.device)
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
            negative_idx = negative_idx.cpu()

        negative_samples = positive[negative_idx]
        negative_ids = positive_ids[negative_idx]
        if self._triple:
            new_samples =  (torch.cat([anchors, positive, negative_samples], dim=1)
                            .resize(sample_num//2*3, embedding_dim))
            new_ids = (torch.stack([anchor_ids, positive_ids, negative_ids], dim=1)
                          .resize(sample_num//2*3))
            new_labels = None
            return new_samples, new_ids, new_labels, SiameseSampleFormat.Triplet
        else:
            new_samples = (torch.cat([anchors, positive, anchors, negative_samples], dim=1)
                           .resize(sample_num * 2, embedding_dim))
            new_ids = (torch.stack([anchor_ids, positive_ids, anchor_ids, negative_ids], dim=1).resize(sample_num * 2))
            new_labels = torch.stack([torch.ones(sample_num // 2), torch.zeros(sample_num // 2)], dim=1).resize(sample_num)
            return new_samples, new_ids, new_labels.to(labels.device), SiameseSampleFormat.Pair

    def __str__(self):
        return f"MarginSampler(margin={self._margin}, triple={self._triple})"
