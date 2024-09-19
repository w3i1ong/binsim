import dgl
import numpy
import torch
from dgl import DGLGraph
from binsim.disassembly.backend.ida import CodeAST
from typing import List, Tuple, Dict, Any
from .datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.nn.layer.dagnn.dagrnn_ops import create_adj_list


class CodeASTSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[CodeAST], sample_id: List[Any]=None, tags=None,
                 with_name=True, **kwargs):
        super().__init__(data, sample_id=sample_id, with_name=with_name, tags=tags, **kwargs)

    @staticmethod
    def collate_fn_py(data: List[Tuple[DGLGraph, torch.Tensor]]) \
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

    @staticmethod
    def collate_fn_raw(data: List[bytes], use_fast=False, **kwargs) -> Any:
        edges, (features, node_nums, callee_num) = CodeAST.collate_raw_neural_input(data, **kwargs)
        root_idx, base = [], 0
        for num in node_nums:
            root_idx.append(base)
            base += num
        root_idx = numpy.array(root_idx)
        if not use_fast:
            graphs = []
            for src_list, dst_list in edges:
                graphs.append(dgl.graph((src_list, dst_list)))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.from_numpy(features), torch.from_numpy(callee_num), torch.from_numpy(root_idx)
        else:
            from binsim.neural.nn.layer.dagnn.utils.utils import prepare_update_information_for_faster_forward
            edges, (features, node_nums, callee_num) = CodeAST.collate_raw_neural_input(data, **kwargs)
            adj_list = create_adj_list(edges, node_nums)
            prop_info = prepare_update_information_for_faster_forward(adj_list)
            # prop_info.check(adj_list) # only used for debug
            return prop_info, torch.from_numpy(features), torch.from_numpy(callee_num), torch.from_numpy(root_idx)


class CodeASTSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[CodeAST]], sample_format: str = None, **kwargs):
        super().__init__(data, sample_format=sample_format, **kwargs)

    @property
    def SingleSampleClass(self):
        return CodeASTSampleDataset
