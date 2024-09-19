from binsim.disassembly.backend.binaryninja import TokenCFG
from typing import List, Tuple, Dict, Any, Iterable
from dgl import DGLGraph
import dgl
import torch
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.lm.ins2vec import Ins2vec


class TokenCFGSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[TokenCFG],
                 samples_id: torch.Tensor=None,
                 with_name=False,
                 ins2vec=None,
                 tags=None, **kwargs):
        super().__init__(data, sample_id=samples_id, with_name=with_name, tags=tags, **kwargs)
        if ins2vec is not None:
            self._ins2idx = Ins2vec.load(ins2vec).ins2idx
        else:
            self._ins2idx = None

    def transform_sample(self, sample: TokenCFG):
        if self._ins2idx is not None:
            sample.replace_tokens(self._ins2idx)
        return super().transform_sample(sample)

    @staticmethod
    def collate_fn_py(data: Iterable[Tuple[DGLGraph, List, List]], **kwargs) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        graphs, tokens, lengths = zip(*data)
        batched_graph = SampleDatasetBase.batch_graph(graphs)
        batched_lengths = []
        for length in lengths:
            batched_lengths.extend(length)
        batched_tokens = []
        for token_list in tokens:
            batched_tokens.extend(token_list)
        max_len = max(batched_lengths)
        batched_lengths = torch.tensor(batched_lengths, dtype=torch.long)
        for token_list in batched_tokens:
            token_list.extend([0] * (max_len - len(token_list)))
        batched_tokens = torch.tensor(batched_tokens, dtype=torch.long)
        return batched_graph, batched_tokens, batched_lengths

    @staticmethod
    def collate_fn_raw(data, **kwargs):
        return TokenCFG.collate_raw_neural_input(data, **kwargs)


class TokenCFGSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[TokenCFG]], sample_format: str = None, ins2vec=None, **kwargs):
        super().__init__(data, sample_format=sample_format, **kwargs)
        if ins2vec is not None:
            self._ins2idx = Ins2vec.load(ins2vec).ins2idx
        else:
            self._ins2idx = None

    def transform_sample(self, sample: TokenCFG | bytes):
        if self._ins2idx is not None and self._neural_input_cache_file is None:
            sample.replace_tokens(self._ins2idx)
        return super().transform_sample(sample)

    @property
    def SingleSampleClass(self):
        return TokenCFGSampleDataset
