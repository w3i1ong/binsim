import torch
from binsim.disassembly.backend.binaryninja import TokenSeq
from typing import List, Tuple, Dict, Any
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.lm.ins2vec import Ins2vec


class TokenSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[TokenSeq],
                 sample_id: List[Any]=None,
                 tags=None,
                 with_name=False,
                 ins2vec=None, **kwargs):
        super().__init__(data, sample_id, with_name=with_name, tags=tags, **kwargs)
        if ins2vec is not None:
            self._ins2idx = Ins2vec.load(ins2vec).ins2idx
        else:
            self._ins2idx = None

    def transform_sample(self, sample: TokenSeq):
        if self._ins2idx is not None:
            sample.replace_tokens(self._ins2idx)
        return super().transform_sample(sample)

    @staticmethod
    def collate_fn_py(data: List[List], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        data = list(data)
        lengths = [len(token_list) for token_list in data]
        max_len = max(max(lengths),1)
        for token_list, length in zip(data, lengths):
            token_list.extend([0] * (max_len - length))
        batched_tokens = torch.tensor(data, dtype=torch.long)
        batched_lengths = torch.tensor(lengths, dtype=torch.long)
        return batched_tokens, batched_lengths

    @staticmethod
    def collate_fn_raw(data, **kwargs):
        return TokenSeq.collate_raw_neural_input(data, **kwargs)

class TokenSeqSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[TokenSeq]],
                 sample_format=None,
                 ins2vec=None, **kwargs):
        super().__init__(data, sample_format=sample_format, **kwargs)
        if ins2vec is not None:
            self._ins2idx = Ins2vec.load(ins2vec)
        else:
            self._ins2idx = None

    def transform_sample(self, sample: TokenSeq):
        if self._ins2idx is not None:
            sample.replace_tokens(self._ins2idx.ins2idx)
        return super().transform_sample(sample)

    @property
    def SingleSampleClass(self):
        return TokenSeqSampleDataset
