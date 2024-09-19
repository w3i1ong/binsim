from abc import ABC, abstractmethod

class ContrastiveLossBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def check(self, sample_format, distance_metric):
        pass

    @abstractmethod
    def __call__(self, samples, labels, pair_sim_func=None, pairwise_sim_func=None):
        pass
