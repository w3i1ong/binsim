from binsim.disassembly.binaryninja import AttributedCFG
from typing import List, Tuple, Dict, Any, Iterable
from dgl import DGLGraph
import dgl
import torch
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase


class ACFGSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[AttributedCFG], sample_id: torch.Tensor=None, tags: List=None, with_name=False, **kwargs):
        super().__init__(data, sample_id=sample_id, with_name=with_name, tags=tags, **kwargs)

    @staticmethod
    def collate_fn(data: Iterable[Tuple[DGLGraph, torch.Tensor]]) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        batched_graph, batched_features = zip(*data)
        batched_graph = SampleDatasetBase.batch_graph(batched_graph)
        batched_features = torch.concatenate(batched_features, dim=0)
        return batched_graph, batched_features.to(torch.float)


class ACFGSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[AttributedCFG]], sample_format=None, **kwargs):
        super().__init__(data, sample_format=sample_format, **kwargs)

    @property
    def SingleSampleClass(self):
        return ACFGSampleDataset
