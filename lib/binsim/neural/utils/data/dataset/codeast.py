import dgl
import torch
from dgl import DGLGraph
from binsim.disassembly.ida import CodeAST
from typing import List, Tuple, Dict, Any
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase


class CodeASTSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[CodeAST], sample_id: torch.Tensor, with_name=True, tags=None):
        super().__init__(data, sample_id=sample_id, with_name=with_name, tags=tags)

    @staticmethod
    def collate_fn(data: List[Tuple[DGLGraph, torch.Tensor]]) \
            -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
        batched_graph, batched_features, callee_num = zip(*data)
        batched_graph = dgl.batch(batched_graph)
        base, root_idx = 0, []
        for node_num in batched_graph.batch_num_nodes():
            root_idx.append(base)
            base += node_num.cpu().item()
        root_idx = torch.tensor(root_idx)
        batched_features = torch.concatenate(batched_features, dim=0)
        callee_num = torch.tensor(callee_num).float()
        return batched_graph, batched_features, callee_num, root_idx


class CodeASTSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[CodeAST]], sample_format: str = None):
        super().__init__(data, sample_format=sample_format)

    @property
    def SingleSampleClass(self):
        return CodeASTSampleDataset
