from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Set
from .utils import get_normalized_arch_name

class NormalizerBase(ABC):
    def __init__(self, arch,):
        self._arch = get_normalized_arch_name(arch)

    @property
    def arch(self):
        return self._arch

    @abstractmethod
    def __call__(self, function):
        pass


class CFGNormalizerBase(NormalizerBase):
    def __init__(self, arch):
        super(CFGNormalizerBase, self).__init__(arch)

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

class BinsimFunction(ABC):
    def __init__(self, function_name, func_hash, func_arch, node_num, ins_num):
        self._function_name = function_name
        self._arch = func_arch
        self._hash = func_hash
        self._node_num = node_num
        self._ins_num = ins_num

    @property
    def name(self) -> str:
        return self._function_name

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def hash(self):
        return self._hash

    @property
    def node_num(self) -> int:
        return self._node_num

    @property
    @abstractmethod
    def features(self):
        pass

    @abstractmethod
    def as_neural_input(self, *args, **kwargs):
        pass

    @abstractmethod
    def as_neural_input_raw(self, *args, **kwargs):
        pass

    def preprocess_neural_input_raw(self, inputs, **kwargs):
        return inputs

    @staticmethod
    @abstractmethod
    def collate_raw_neural_input(inputs: List[bytes], **kwargs):
        pass

    def __len__(self):
        return self._node_num

    @abstractmethod
    def minimize(self):
        pass

class BinsimCFG(BinsimFunction):
    def __init__(self, function_name: str, func_hash: str, func_arch, ins_num,  *,
                 adj_list: Dict[int, List[int]], features):
        super().__init__(function_name, func_hash, func_arch, len(adj_list), ins_num=ins_num)
        assert len(adj_list) == len(features), (f"Expect the number of nodes in adj_list({len(adj_list)}) "
                                                f"and features({len(features)} to be the same.")
        idx2addr, addr2idx, adj_list = self._normalize_node_addr(adj_list)
        self._idx2addr = idx2addr
        self._adj_list = adj_list
        self._features = [features[addr] for addr in idx2addr]
        self._entries = self.__get_entries(adj_list)

    @staticmethod
    def __get_entries(adj_list):
        in_degree = {addr: 0 for addr in adj_list}
        for src, dst_list in adj_list.items():
            for dst in dst_list:
                in_degree[dst] += 1
        result = set()
        for addr, degree in in_degree.items():
            if degree == 0:
                result.add(addr)
        if len(result) == 0:
            result.add(0)
        return result

    def minimize(self):
        self._idx2addr = None
        self._adj_list = None
        self._features = None
        self._entries = None

    @staticmethod
    def _normalize_node_addr(adj_list):
        address = sorted(list(adj_list.keys()))
        addr2idx = {addr: idx for idx, addr in enumerate(address)}
        new_adj_list = {}
        for src, dst_list in adj_list.items():
            new_adj_list[addr2idx[src]] = [addr2idx[dst] for dst in dst_list]
        return address, addr2idx, new_adj_list

    @property
    def features(self):
        return self._features

    @abstractmethod
    def as_neural_input(self, *args, **kwargs):
        pass

    @abstractmethod
    def as_neural_input_raw(self, *args, **kwargs)->bytes:
        pass

    @staticmethod
    @abstractmethod
    def collate_raw_neural_input(inputs:List[bytes], **kwargs):
        pass

    @property
    def entries(self) -> Set[int]:
        return self._entries

    @property
    def adj_list(self) -> Dict[int, List[int]]:
        return self._adj_list

    @property
    def basic_block_address(self) -> List[int]:
        return [i for i in range(len(self._idx2addr))]

    @property
    def original_address(self):
        return self._idx2addr

    def prepare_graph_info(self, expand_time=0)->Tuple[List, List, int, List]:
        from binsim.utils import fastGraph

        if expand_time is not None:
            # expand_time is not None, loop expansion should be applied.
            # 1. first we reconstruct the original graph
            edge_src_list, edge_dst_list = [], []
            for src, dst_list in self.adj_list.items():
                edge_src_list.extend([src+1] * len(dst_list))
                edge_dst_list.extend([dst+1 for dst in dst_list])

            # To deal with the case that there are multiple entries in the CFG, we add a dummy node.
            edge_src_list.extend([0] * len(self.entries))
            edge_dst_list.extend([entry+1 for entry in self.entries])

            # 2. build our fastGraph.
            cfg = fastGraph(self.node_num + 1, edge_src_list, edge_dst_list, 0)
            nodeId, (U, V) = cfg.toDAG(k=expand_time)
            # remember to remove the dummy node
            U, V, nodeId = np.array(U), np.array(V), np.array(nodeId)
            valid_index = (U != 0)
            U, V, node_num, nodeId = U[valid_index] - 1, V[valid_index] - 1, len(nodeId) - 1, nodeId[1:] - 1
            return list(U), list(V), node_num, list(nodeId)
        else:
            edge_src_list, edge_dst_list = [], []
            for src, dst_list in self.adj_list.items():
                edge_src_list.extend([src] * len(dst_list))
                edge_dst_list.extend([dst for dst in dst_list])
            return edge_src_list, edge_dst_list, self.node_num, list(range(self.node_num))
