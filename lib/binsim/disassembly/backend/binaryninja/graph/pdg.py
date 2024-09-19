import torch
import logging
import typing
import numpy as np
from binsim.disassembly.backend.binaryninja.utils import compute_function_hash
from binsim.disassembly.core import BinsimFunction, CFGNormalizerBase
from typing import Dict, List, Set, Tuple, TYPE_CHECKING
from scipy.sparse.csgraph import floyd_warshall
from ..utils import control_reachable_distance_matrix
from .TokenCFG import TokenCFGMode
from .instruction import BSInsOperandType, BSInstruction
from .InsCFG import InsCFGNormalizer
from binsim.utils import init_logger

if typing.TYPE_CHECKING:
    from binaryninja import Function, BasicBlock

logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)


class ProgramDependencyGraph(BinsimFunction):
    def __init__(self, function_name: str,
                 arch: str,
                 basic_blocks: Dict[int, List[BSInstruction]],
                 control_flow_edges: List[Tuple[int, int]],
                 func_hash: str):
        """
        :param function_name: The name of the function.
        :param basic_blocks: A dictionary of instructions. The key is the address of the instruction and the value is
        the instruction. Each instruction is a PDGInsInstruction.
        :param control_flow_edges: A list of edges that represent the control flow. Each edge is a tuple of two integers.
            The first integer is the source node and the second integer is the destination node.
        """
        super().__init__(function_name, func_hash=func_hash, func_arch=arch)
        self._features = basic_blocks
        self._adj_list = control_flow_edges
        self._mode = TokenCFGMode.TOKEN
        self._normalize()
        self._relative_distance_matrix = None

    def _normalize(self):
        """
        Assign a unique index to each instruction and replace their address with the index.
        :return:
        """
        addr2idx = {addr: idx for idx, addr in enumerate(self.basic_block_address)}
        self._features = [self._features[addr] for addr in self.basic_block_address]
        self._adj_list = [(addr2idx[src], addr2idx[dst]) for src, dst in self._adj_list]

    @property
    def control_flow_reachable_matrix(self) -> torch.Tensor:
        # calculate the number of instructions in each basic block
        distance_matrix = control_reachable_distance_matrix(self._features, self._adj_list)
        return torch.from_numpy(distance_matrix)

    @staticmethod
    def _calculate_distance_matrix(adj_matrix: np.array):
        distance_matrix = floyd_warshall(adj_matrix)
        return distance_matrix

    def as_neural_input(self, with_control_flow=True) -> Tuple[List[BSInstruction], torch.Tensor]:
        control_flow = self.control_flow_reachable_matrix if with_control_flow else None
        features = []
        for ins_seq in self._features:
            features.extend(ins_seq)
        return features, control_flow

    def __repr__(self):
        return f"<ProgramDependencyGraph::{self.name}" \
               f"(nodes={len(self.features)})>"

    def unique_tokens(self) -> Set[str]:
        assert self._mode == 'token', "The CFG is not in token mode."
        tokens = set()
        for bb in self._features:
            for ins in bb:
                tokens.add(ins.mnemonic)
                for op in ins.operands:
                    if op.type == BSInsOperandType.REG:
                        tokens.add(op.value)
                    elif op.type == BSInsOperandType.IMM:
                        continue
                    elif op.type == BSInsOperandType.ARM_MEM:
                        base, index, shift, disp, flag = op.value
                        tokens.add(base)
                        tokens.add(index)
                        tokens.add(flag)
                    elif op.type == BSInsOperandType.X86_MEM:
                        base, index, scale, disp = op.value
                        tokens.add(base)
                        tokens.add(index)
                    elif op.type == BSInsOperandType.MIPS_MEM:
                        base, index, disp = op.value
                        tokens.add(base)
                        tokens.add(index)
                    else:
                        raise ValueError(f"Unknown operand type: {op.type}")
        return tokens

    def replace_tokens(self, token2id):
        if self._mode != TokenCFGMode.TOKEN:
            return
        for bb in self._features:
            for ins in bb:
                ins.mnemonic = token2id[ins.mnemonic]
                for operand in ins.operands:
                    if operand.type == BSInsOperandType.REG:
                        if isinstance(operand.value, str):
                            operand.value = token2id[operand.value]
                    elif operand.type == BSInsOperandType.IMM:
                        continue
                    elif operand.type == BSInsOperandType.ARM_MEM:
                        base, index, shift, disp, flag = operand.value
                        operand.value = (token2id[base], token2id[index], shift, disp, token2id[flag])
                    elif operand.type == BSInsOperandType.X86_MEM:
                        base, index, scale, disp = operand.value
                        operand.value = (token2id[base], token2id[index], scale, disp)
                    elif operand.type == BSInsOperandType.MIPS_MEM:
                        base, index, disp = operand.value
                        operand.value = (token2id[base], token2id[index], disp)
                    else:
                        raise ValueError(f"Unknown operand type: {operand.type}.")
        self._mode = TokenCFGMode.ID


class PDGNormalizer(CFGNormalizerBase):
    SUPPORED_ARCH = ['x86_64', 'x86', 'arm', 'arm64', 'mips32', 'mips64']

    def __init__(self, arch):
        super().__init__(arch=arch)
        if not self.is_arch_supported(self.arch):
            raise KeyError(f"Unknown architecture: {self.arch}")
        self._extract_basic_block = getattr(self, f'_extract_basic_block_{self.arch}')

    def is_arch_supported(self, arch):
        return arch in self.SUPPORED_ARCH

    def __call__(self, function: 'Function') -> ProgramDependencyGraph:
        """
        Given a function, extract the control flow graph, instruction list and data flow graph.
        :param function: A function object.
        :return:
        """
        control_flow_edges = self._extract_control_flow(function)
        basic_blocks = self._extract_instructions(function)
        return ProgramDependencyGraph(function.name,
                                      self.arch,
                                      basic_blocks,
                                      control_flow_edges,
                                      func_hash=compute_function_hash(function))

    @staticmethod
    def _extract_control_flow(function: 'Function') -> List[Tuple[int, int]]:
        """
        Given a function, extract the control flow transfer edges between basic blocks and
            instruction addresses in each basic block.
        :param function: A function object from binaryninja.
        :return: edges, basic_blocks. edges is a list of tuples, each tuple (u,v) means there exists a control transfer
        from basic block u to node v. basic_blocks is a list, each list in it contains the address of instructions in
        the basic block.
        """
        edges = []
        for basic_block in function:
            # collect edges between basic blocks
            for outgoing_edge in basic_block.outgoing_edges:
                edges.append((basic_block.start, outgoing_edge.target.start))
        return edges

    def _extract_instructions(self, function: 'Function') -> Dict[int, List[BSInstruction]]:
        pdg_basic_blocks = {}
        for basic_block in function.basic_blocks:
            basic_block_ins = self._extract_basic_block
            pdg_basic_blocks[basic_block.start] = basic_block_ins
        return pdg_basic_blocks

    @staticmethod
    def _extract_basic_block_x86(basic_block: 'BasicBlock') -> List[BSInstruction]:
        return InsCFGNormalizer.extract_bb_tokens_x86(basic_block)

    @staticmethod
    def _extract_basic_block_arm(basic_block: 'BasicBlock') -> List[BSInstruction]:
        return InsCFGNormalizer.extract_bb_tokens_arm(basic_block)

    @staticmethod
    def _extract_basic_block_arm64(basic_block: 'BasicBlock') -> List[BSInstruction]:
        return InsCFGNormalizer.extract_bb_tokens_arm64(basic_block)

    @staticmethod
    def _extract_basic_block_x86_64(basic_block: 'BasicBlock') -> List[BSInstruction]:
        return InsCFGNormalizer.extract_bb_tokens_x86_64(basic_block)

    @staticmethod
    def _extract_basic_block_mips32(basic_block: 'BasicBlock') -> List[BSInstruction]:
        return InsCFGNormalizer.extract_bb_tokens_mips32(basic_block)

    @staticmethod
    def _extract_basic_block_mips64(basic_block: 'BasicBlock') -> List[BSInstruction]:
        return InsCFGNormalizer.extract_bb_tokens_mips64(basic_block)
