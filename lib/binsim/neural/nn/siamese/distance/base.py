import torch
from abc import abstractmethod, ABC

class DistanceBase(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def pairwise_similarity(self, x, y):
        pass

    @abstractmethod
    def similarity(self, x, y):
        pass

    @abstractmethod
    def pairwise_similarity_for_search(self, x, y):
        pass

    @abstractmethod
    def similarity_for_search(self, x, y):
        pass

    @property
    @abstractmethod
    def metric(self):
        raise NotImplementedError("metric property not implemented")
