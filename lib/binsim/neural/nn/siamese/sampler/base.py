from abc import ABC, abstractmethod

class SamplerBase(ABC):
    @abstractmethod
    def __call__(self, embeddings, labels, sample_ids, pairwise_distance, sample_format, distance_metric):
        pass
