from enum import Enum


class SiameseSampleFormat(Enum):
    Pair = 'pair'
    SemiHardPair = 'semi-hard-pair'
    Triplet = 'triplet'
    SemiHardTriplet = 'semi-hard-triplet'
    Proxy = 'proxy'
    InfoNCESamples = 'info-nce-samples'

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return self.value


class SiameseMetric:
    ClassifyMetric = {'auc', 'roc'}
    SearchMetric = {'mrr', 'hit', 'recall', 'precision', 'ndcg'}
    def __init__(self, metric_str: str):
        self._metric_str = metric_str
        self._is_search_metric = False
        self._top_value = -1
        self.__init(metric_str)

    def __init(self, metric_str):
        if '@' in metric_str:
            self._is_search_metric = True
            self._metric_str, self._top_value = metric_str.split('@')
            self._metric_str = self._metric_str.lower()
            self._top_value = int(self._top_value)
            if self._metric_str not in SiameseMetric.SearchMetric or self._top_value <= 0:
                raise ValueError(f"Invalid search metric: {self}")
        else:
            self._metric_str = metric_str.lower()
            self._is_search_metric = False
            if self._metric_str not in SiameseMetric.ClassifyMetric:
                raise ValueError(f"Invalid classification metric: {self}")

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
    @property
    def AUC(cls):
        return SiameseMetric('AUC')

    @classmethod
    @property
    def ROC(cls):
        return SiameseMetric('ROC')

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

