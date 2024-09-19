import dgl
import torch
from dgl import DGLGraph
from typing import List, Tuple, Dict, Any, Iterable
from binsim.disassembly.backend.binaryninja import AttributedCFG
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase

__all__ = ["ACFGSampleDataset", "ACFGSamplePairDataset"]

class ACFGSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[AttributedCFG], sample_id: torch.Tensor=None, tags: List=None, with_name=False, **kwargs):
        super().__init__(data, sample_id=sample_id, with_name=with_name, tags=tags, **kwargs)

    @staticmethod
    def collate_fn_py(data: Iterable[Tuple[DGLGraph, torch.Tensor]], **kwargs) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        batched_graph, batched_features = zip(*data)
        batched_graph = SampleDatasetBase.batch_graph(batched_graph)
        batched_features = torch.concatenate(batched_features, dim=0)
        return batched_graph, batched_features.to(torch.float)


    @staticmethod
    def collate_fn_raw(data: list[bytes], **kwargs) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        graph, features = AttributedCFG.collate_raw_neural_input(data)
        return graph, features.to(torch.float)

class ACFGSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[AttributedCFG]], sample_format=None, **kwargs):
        super().__init__(data, sample_format=sample_format, **kwargs)

    @property
    def SingleSampleClass(self):
        return ACFGSampleDataset
