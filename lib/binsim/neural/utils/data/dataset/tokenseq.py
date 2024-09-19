import torch
from binsim.disassembly.binaryninja import TokenCFG, TokenCFGDataForm
from typing import List, Tuple, Dict, Any
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.lm.ins2vec import Ins2vec


class InsStrSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[TokenCFG],
                 sample_id: List[Any]=None,
                 tags=None,
                 with_name=False,
                 ins2vec=None):
        for cfg in data:
            cfg.data_form = TokenCFGDataForm.InsStrSeq
        super().__init__(data, sample_id, with_name=with_name, tags=tags)
        if ins2vec is not None:
            self._ins2idx = Ins2vec.load(ins2vec).ins2idx
        else:
            self._ins2idx = None

    def transform_sample(self, sample: TokenCFG):
        if self._ins2idx is not None:
            sample.replace_tokens(self._ins2idx)
        return super().transform_sample(sample)

    @staticmethod
    def collate_fn(data: List[List]) -> Tuple[torch.Tensor, torch.Tensor]:
        data = list(data)
        lengths = [len(token_list) for token_list in data]
        max_len = max(max(lengths),1)
        for token_list, length in zip(data, lengths):
            token_list.extend([0] * (max_len - length))
        batched_tokens = torch.tensor(data, dtype=torch.long)
        batched_lengths = torch.tensor(lengths, dtype=torch.long)
        return batched_tokens, batched_lengths


class InsStrSeqSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[TokenCFG]],
                 sample_format=None,
                 ins2vec=None):
        for k, v in data.items():
            for cfg in v:
                cfg.data_form = TokenCFGDataForm.InsStrSeq
        super().__init__(data, sample_format=sample_format)
        if ins2vec is not None:
            self._ins2idx = Ins2vec.load(ins2vec)
        else:
            self._ins2idx = None

    def transform_sample(self, sample: TokenCFG):
        if self._ins2idx is not None:
            sample.replace_tokens(self._ins2idx.ins2idx)
        return super().transform_sample(sample)

    @property
    def SingleSampleClass(self):
        return InsStrSeqSampleDataset
