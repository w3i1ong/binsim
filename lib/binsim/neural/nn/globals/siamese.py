from enum import Enum


class SiameseSampleFormat(Enum):
    Pair = 'pair'
    PositivePair = "positive-pair"
    PositivePairSplit = "positive-pair-split"
    Triplet = 'triplet'
    QueryTarget = 'query-target'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return self.value

class EmbeddingDistanceMetric(Enum):
    Cosine = 'cosine'
    Euclidean = 'euclidean'
    SelfDefined = 'self-defined'
    AsteriaDistance = 'asteria'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return self.value


class SiameseMetric:
    SearchMetric = {'mrr', 'hit', 'recall', 'precision', 'ndcg'}
    def __init__(self, metric_str: str):
        self._metric_str = metric_str
        self._is_search_metric = False
        self._top_value = -1
        self.__init(metric_str)

    def __init(self, metric_str):
        assert '@' in metric_str
        self._is_search_metric = True
        self._metric_str, self._top_value = metric_str.split('@')
        self._metric_str = self._metric_str.lower()
        self._top_value = int(self._top_value)
        if self._metric_str not in SiameseMetric.SearchMetric or self._top_value <= 0:
            raise ValueError(f"Invalid search metric: {self}")

    def is_search_metric(self):
        return self._is_search_metric

    @property
    def name(self):
        return self._metric_str

    @property
    def top_value(self):
        if not self._is_search_metric:
            raise ValueError(f"Metric {self} is not a search metric, and you cannot get top value.")
        return self._top_value

    def __repr__(self):
        if self._is_search_metric:
            return f"{self._metric_str}@{self._top_value}"
        else:
            return self._metric_str

    def __str__(self):
        return self.__repr__()

    def __format__(self, format_spec):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    @classmethod
    def MRR(cls, k):
        return SiameseMetric(f'MRR@{k}')

    @classmethod
    def Hit(cls, k):
        return SiameseMetric(f'Hit@{k}')

    @classmethod
    def Recall(cls, k):
        return SiameseMetric(f'Recall@{k}')

    @classmethod
    def Precision(cls, k):
        return SiameseMetric(f'Precision@{k}')

    @classmethod
    def nDCG(cls, k):
        return SiameseMetric(f'nDCG@{k}')

    def __eq__(self, other):
        assert isinstance(other, SiameseMetric)
        return self.__repr__() == other.__repr__()

