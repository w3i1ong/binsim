from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from .utils import get_normalized_arch_name
import pickle

class NormalizerBase(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, function):
        pass


class CFGNormalizerBase(NormalizerBase):
    def __init__(self, arch):
        super(CFGNormalizerBase, self).__init__()
        self._arch = get_normalized_arch_name(arch)

    @property
    def arch(self):
        return self._arch

    @abstractmethod
    def __call__(self, function):
        pass

    @staticmethod
    def extract_adj_list(function) -> Tuple[Dict[int, List[int]], List]:
        adj_list, entrypoints = {}, []
        for basic_block in function.basic_blocks:
            adj_list[basic_block.start] = []
            for edge in basic_block.outgoing_edges:
                adj_list[basic_block.start].append(edge.target.start)
            if len(basic_block.incoming_edges) == 0:
                entrypoints.append(basic_block.start)
        if len(entrypoints) == 0:
            entrypoints.append(function.start)
        return adj_list, entrypoints


class MultiNormalizer(NormalizerBase):
    def __init__(self, normalizers: List[NormalizerBase]):
        super(MultiNormalizer, self).__init__()
        self.normalizers = normalizers

    def __add__(self, other):
        if isinstance(other, MultiNormalizer):
            self.normalizers = self.normalizers + other.normalizers
        elif isinstance(other, NormalizerBase):
            self.normalizers = self.normalizers + [other]
        else:
            raise TypeError("Cannot add MultiNormalizer to {}".format(type(other)))

    def __radd__(self, other):
        if isinstance(other, MultiNormalizer):
            self.normalizers = other.normalizers + self.normalizers
        elif isinstance(other, NormalizerBase):
            self.normalizers = [other] + self.normalizers
        else:
            raise TypeError("Cannot add {} to MultiNormalizer".format(type(other)))

    def __call__(self, function):
        for normalizer in self.normalizers:
            normalizer(function)


class CFGBase(ABC):
    def __init__(self, function_name, func_hash, func_arch, node_num, entries=None):
        self._function_name = function_name
        self._adj_list = {}
        self._features = {}
        self._arch = func_arch
        self._hash = func_hash
        self._entries = entries
        self._node_num = node_num

    @property
    def hash(self):
        return self._hash

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def name(self) -> str:
        return self._function_name

    @property
    def features(self) -> Dict[int, Any]:
        return self._features

    @property
    def node_num(self) -> int:
        return self._node_num

    @property
    def adj_list(self) -> Dict[int, List[int]]:
        return self._adj_list

    @property
    def entries(self) -> List[int]:
        if self._entries is None:
            entries = set(self.adj_list.keys())
            for _, next_nodes in self._adj_list.items():
                entries -= set(next_nodes)
            res = list(entries)
            if len(entries) == 0:
                res = [min(self.adj_list.keys())]
            self._entries = res
        return self._entries

    @property
    def basic_block_address(self) -> List[int]:
        return sorted(list(self._features.keys()))

    @abstractmethod
    def as_neural_input(self, *args, **kwargs):
        pass

    def save(self, file: str):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: str):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.features)

    def as_dgl_graph(self, expand_time=0):
        import torch
        from dgl import graph
        from binsim.utils import fastGraph

        if expand_time is not None:
            # expand_time is not None, loop expansion should be applied.
            # 1. first we reconstruct the original graph
            addr2idx = {addr: idx for idx, addr in enumerate(self.basic_block_address, start=1)}
            edge_src_list, edge_dst_list = [], []
            node_num = len(addr2idx)
            for src, dsts in self.adj_list.items():
                edge_src_list.extend([addr2idx[src]] * len(dsts))
                edge_dst_list.extend([addr2idx[dst] for dst in dsts])

            # To deal with the case that there are multiple entries in the CFG, we add a dummy node.
            edge_src_list.extend([0] * len(self.entries))
            edge_dst_list.extend([addr2idx[entry] for entry in self.entries])

            # 2. build our fastGraph.
            cfg = fastGraph(node_num + 1, edge_src_list, edge_dst_list, 0)
            nodeId, (U, V) = cfg.toDAG(k=expand_time)
            # remember to remove the dummy node
            U, V = torch.tensor(U), torch.tensor(V)
            valid_index = (U != 0)
            cfg = graph((U[valid_index] - 1, V[valid_index] - 1), num_nodes=len(nodeId) - 1)
            cfg.ndata['nodeId'] = torch.tensor(nodeId[1:]) - 1
            idx2addr = {idx-1: addr for addr, idx in addr2idx.items()}
        else:
            addr2idx = {addr: idx for idx, addr in enumerate(self.basic_block_address, start=0)}
            edge_src_list, edge_dst_list = [], []
            for src, dsts in self.adj_list.items():
                edge_src_list.extend([addr2idx[src]] * len(dsts))
                edge_dst_list.extend([addr2idx[dst] for dst in dsts])
            cfg = graph((edge_src_list, edge_dst_list), num_nodes=len(addr2idx))
            idx2addr = {idx: addr for addr, idx in addr2idx.items()}
        return cfg, idx2addr
