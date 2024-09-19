import os
import logging
import numpy as np
import capstone as cs
from binsim.utils import init_logger
from typing import Dict, List, TYPE_CHECKING
from binsim.disassembly.core import Architecture
from binsim.disassembly.backend.binaryninja.utils import *
from binsim.disassembly.core import CFGNormalizerBase, BinsimCFG
from .datautils import acfg_pack_neural_input, acfg_collate_neural_input
logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)

if TYPE_CHECKING:
    from binaryninja import Function, BasicBlock

class AttributedCFG(BinsimCFG):
    def __init__(self, function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List],
                 func_hash: str,
                 func_arch: str,
                 ins_num: int):
        """
        Build an attributed CFG from the given adjacency list and features.
        :param function_name: The name of the function.
        :param adj_list: The adjacency list of the CFG.
        :param features: The features of each basic block.
        """
        super(AttributedCFG, self).__init__(function_name, func_hash, func_arch,
                                            adj_list=adj_list, features=features, ins_num=ins_num)

    def as_neural_input(self, expand_time=None):
        """
        Convert the CFG to suitable input for neural network, including DGLGraph and features tensor.
        :return: A tuple of DGLGraph and features tensor.
        """
        import dgl, torch
        src, dst, node_num, nodeId = self.prepare_graph_info(expand_time=expand_time)
        graph: dgl.DGLGraph = dgl.graph((src, dst), num_nodes=node_num)
        if nodeId is None:
            graph.ndata['nodeId'] = torch.from_numpy(nodeId)
        features: torch.Tensor = torch.from_numpy(np.stack(self.features))
        return graph, features

    def as_neural_input_raw(self, expand_time=None)->bytes:
        assert expand_time is None, "expand_time is not supported in raw input."
        src, dst, node_num, node_id = self.prepare_graph_info(expand_time=None)
        return acfg_pack_neural_input(self.node_num, src, dst, self.features)

    @staticmethod
    def collate_raw_neural_input(inputs:List[bytes], **kwargs):
        import dgl, torch
        edges, features, node_nums = acfg_collate_neural_input(inputs)
        graphs = []
        for (src, dst), node_num in zip(edges, node_nums):
            graph = dgl.graph((src.astype(np.int32), dst.astype(np.int32)), num_nodes=node_num)
            graphs.append(graph)
        graph: dgl.DGLGraph = dgl.batch(graphs)
        return graph, torch.from_numpy(features)


    def __repr__(self):
        if self.features is not None:
            return f"<AttributedCFG::{self.name}" \
                   f"(nodes={self.node_num}, features={len(self.features[0])})>"
        else:
            return f"<AttributedCFG::{self.name}" \
                   f"(nodes={self.node_num})>"

    def minimize(self):
        super().minimize()
        return self


class AttributedCFGNormalizer(CFGNormalizerBase):
    INS_TYPE = ['arith', 'call', 'logic', 'transfer']
    SUPPORTED_ARCH = [Architecture.X64, Architecture.X86, Architecture.ARM32, Architecture.ARM64,
                      Architecture.MIPS32, Architecture.MIPS64]
    ARCH2CSARCH = {
        Architecture.X64: cs.CS_ARCH_X86,
        Architecture.X86: cs.CS_ARCH_X86,
        Architecture.ARM32: cs.CS_ARCH_ARM,
        Architecture.ARM64: cs.CS_ARCH_ARM64,
        Architecture.MIPS32: cs.CS_ARCH_MIPS,
        Architecture.MIPS64: cs.CS_ARCH_MIPS,
    }

    ARCH2CSMODE = {
        Architecture.X86: cs.CS_MODE_32,
        Architecture.X64: cs.CS_MODE_64,
        Architecture.ARM32: cs.CS_MODE_ARM,
        Architecture.ARM64: cs.CS_MODE_ARM,
        Architecture.MIPS32: cs.CS_MODE_MIPS32,
        Architecture.MIPS64: cs.CS_MODE_MIPS64,
    }

    def __init__(self, arch):
        super(AttributedCFGNormalizer, self).__init__(arch=arch)
        self._ins2type: dict[str, str] = {}
        self._unknown_ins = set()
        self._init_ins2type()
        self._cs = cs.Cs(mode=self.ARCH2CSMODE[self.arch],
                         arch=self.ARCH2CSARCH[self.arch])
        self._cs.detail = True

    @property
    def arch(self):
        return self._arch

    @property
    def simple_arch(self):
        match self.arch:
            case Architecture.X64 | Architecture.X86:
                return 'x86'
            case Architecture.ARM32 | Architecture.ARM64:
                return 'arm'
            case Architecture.MIPS32 | Architecture.MIPS64:
                return 'mips'
        raise KeyError(f"Unknown architecture: {self.arch}")

    def is_arch_supported(self, arch):
        return arch in self.SUPPORTED_ARCH

    @property
    def ins2type(self):
        return self._ins2type

    def _init_ins2type(self):
        if not self.is_arch_supported(self.arch):
            raise KeyError(f"Unknown architecture: {self.arch}")
        cur_dir = os.path.split(__file__)[0]
        data_dir = f'{cur_dir}/data/mnem/{self.simple_arch}'
        for ins_type in self.INS_TYPE:
            with open(f'{data_dir}/{ins_type}', 'r') as f:
                for instruction in list(map(lambda x: x.strip(), f.readlines())):
                    self.ins2type[instruction] = ins_type

    def __call__(self, function: "Function") -> AttributedCFG:
        adj_list, entry_points = self.extract_adj_list(function)
        features, instruction_num = {}, 0
        for basic_block in function.basic_blocks:
            features[basic_block.start] = self.__extract_bb_features(basic_block)
            instruction_num += basic_block.instruction_count
        graph_features = self.__extract_graph_features(adj_list)
        for key in adj_list.keys():
            features[key] = features[key] + graph_features[key]
        return AttributedCFG(function_name=function.name,
                             adj_list=adj_list,
                             features=features,
                             func_hash=compute_function_hash(function),
                             func_arch=function.arch.name,
                             ins_num=instruction_num)

    def __extract_bb_features(self, basic_block: "BasicBlock") -> List[int]:
        string_ref = 0  # number of global variable references
        imm_operand = 0  # number of immediate operands

        # count the number of immediate operands
        for ins in self._cs.disasm(basic_block.view.read(basic_block.start, basic_block.length), basic_block.start):
            for operand in ins.operands:
                if operand.type == cs.CS_OP_IMM:
                    imm_operand += 1

        # count the number of data references as string references
        for instruction in basic_block.disassembly_text:
            if "data_" in instruction.__repr__():
                string_ref += 1

        # count the number of different types of instructions
        statistics = {}
        for tokens, _ in basic_block:
            op = tokens[0].text
            op_type = self.ins2type.get(op, None)
            if op_type is not None:
                statistics[op_type] = statistics.get(op_type, 0) + 1
        res = [statistics.get(ins_type, 0) for ins_type in self.INS_TYPE]

        res.extend([string_ref, imm_operand, basic_block.instruction_count])
        return res

    @staticmethod
    def __extract_graph_features(adj_list) -> Dict[int, List]:
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_nodes_from(adj_list.keys())
        for key, successors in adj_list.items():
            graph.add_edges_from([(key, successor) for successor in successors])
        features = nx.betweenness_centrality(graph)
        for key, successors in adj_list.items():
            features[key] = [features[key], len(successors)]
        return features
