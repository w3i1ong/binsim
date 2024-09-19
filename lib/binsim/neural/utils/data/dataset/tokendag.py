from torch import Tensor
import numpy as np
from binsim.disassembly.binaryninja import TokenCFG, TokenCFGDataForm
from typing import List, Tuple, Dict, Any, Generator, Union
from dgl import DGLGraph, DGLHeteroGraph
import dgl
import torch
import pickle
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from itertools import chain


class TokenDAGSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[TokenCFG],
                 func_id: torch.Tensor,
                 with_name=False,
                 expand_time=0):
        data = [(cfg, id) for cfg, id in zip(data, func_id) if cfg.isReducible()]
        func_id = torch.tensor([id for cfg, id in data])
        data = [cfg for cfg, id in data]
        for cfg in data:
            cfg.data_form = TokenCFGDataForm.InsStrGraph
        super().__init__(data, sample_id=func_id, to_dag=True, with_name=with_name, k=expand_time)

    @staticmethod
    def collate_fn(data: List[Tuple[DGLGraph, List, List]]) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        graphs, tokens, lengths = zip(*data)

        base = 0
        for graph, length in zip(graphs, lengths):
            graph.ndata['nodeId'] += base
            base += len(length)

        batched_graph = dgl.batch(graphs)
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


class TokenDAGSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[TokenCFG]],
                 sample_format: str = None,
                 expand_time=0):
        for k in list(data.keys()):
            v = [cfg for cfg in data[k] if cfg.isReducible()]
            for cfg in v:
                cfg.data_form=TokenCFGDataForm.InsStrGraph
            data[k] = v
            if len(v) == 0:
                del data[k]
        super().__init__(data, to_dag=True, k=expand_time, sample_format=sample_format)

    @property
    def SingleSampleClass(self):
        return TokenDAGSamplePairDataset
