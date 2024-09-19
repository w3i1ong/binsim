import dgl
import logging
import typing
import numpy as np
from enum import Enum
from typing import Any
from dgl import DGLGraph
from binsim.utils import init_logger
from typing import Dict, List, Set, Tuple, Union
from binsim.disassembly.backend.binaryninja.utils import compute_function_hash
from binsim.disassembly.core import BinsimFunction, CFGNormalizerBase
logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)

if typing.TYPE_CHECKING:
    from binaryninja import Function, BasicBlock

class MneCFGMode(Enum):
    TEXT = 0
    ID = 1


class MnemonicCFG(BinsimFunction):
    def __init__(self,
                 function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List[List[str]]],
                 func_hash: str,
                 func_arch: str,
                 entry_points=None):
        """
        Build a token CFG from the given adjacency list and features.
        :param function_name: The name of the function.
        :param adj_list: The adjacency list of the CFG.
        :param features: A list, in which each element is a list of tokens.
        :param func_hash: The hash of the function, used to remove duplicate functions.
        :param func_arch: The architecture of the function.
        """
        super(MnemonicCFG, self).__init__(function_name, func_hash=func_hash, func_arch=func_arch,
                                          entries=entry_points, node_num=len(adj_list))
        self._adj_list = adj_list
        self._features: Dict[int, List[Union[List[str], int, str]]] = features
        self._mode = MneCFGMode.TEXT

    def unique_tokens(self) -> Set[Union[str, int]]:
        """
        Get the all unique tokens in the CFG.
        :return: A set of unique tokens.
        """
        result = set()
        for basic_block in self.features.values():
            result.update(basic_block)
        return result

    # this function is not necessary, but for the compatibility with the TokenCFGDataForm
    def set_state(self, *args, **kwargs):
        pass

    def replace_tokens(self, token2id: Dict[str, int]):
        """
        Replace tokens in the CFG with token ids.
        :param token2id: A dictionary, which maps tokens to token ids.
        :return:
        """
        if self._mode == MneCFGMode.ID:
            return
        for address, token_list in self.features.items():
            self._features[address] = [token2id.get(token, 0) for instruction in token_list for token in
                                       instruction]
        self._mode = MneCFGMode.ID
        return self

    def as_neural_input(self) -> Tuple[DGLGraph, List, List]:
        """
        Convert the CFG to a DGLGraph and two tensors.
        The first tensor is a block_num * max_token_num tensor, which represents the instruction sequences in each basic
        block.
        The second tensor is a block_num tensor, which represents
        the length of instruction sequence in each basic block.

        :return:
        """
        assert self._mode == MneCFGMode.ID, "Please replace tokens with token ids first!"
        basic_block_address = self.basic_block_address
        address2id = dict(zip(self.basic_block_address, range(len(basic_block_address))))
        # initialize graph
        U, V = [], []
        for src, dst_list in self.adj_list.items():
            U.extend([address2id[src]] * len(dst_list))
            V.extend([address2id[dst] for dst in dst_list])

        # initialize token sequences and their lengths
        token_seq_list: List[Any] = [None] * len(address2id)
        lengths = []
        for addr, addr_id in address2id.items():
            token_seq_list[addr_id] = self._features[addr][:]
            lengths.append(len(self._features[addr]))

        graph = dgl.graph((U, V), num_nodes=len(self.features))
        return graph, token_seq_list, lengths

    def __repr__(self):
        return f'<MnemonicCFG::{self.name}' \
               f'(node_num={len(self._features)})>'


class MnemonicCFGNormalizer(CFGNormalizerBase):
    def __init__(self, arch):
        super(MnemonicCFGNormalizer, self).__init__(arch=arch)

    def __call__(self, function: 'Function') -> MnemonicCFG:
        adj_list, entry_points = self.extract_adj_list(function)
        token_sequences = self._extract_tokens(function)
        cfg = MnemonicCFG(function.name, adj_list, token_sequences,
                          func_hash=compute_function_hash(function),
                          func_arch=self.arch,
                          entry_points=entry_points)
        return cfg

    def _extract_tokens(self, function: 'Function') -> Dict[int, np.array]:
        bb_tokens = {}
        for basic_block in function.basic_blocks:
            bb_tokens[basic_block.start] = self.extract_mnemonics(basic_block)
        return bb_tokens

    @staticmethod
    def extract_mnemonics(basic_block: 'BasicBlock') -> List[str]:
        from binaryninja import InstructionTextTokenType
        mnemonics = []
        for instruction, inst_size in basic_block:
            for token in instruction:
                if token.type == InstructionTextTokenType.InstructionToken:
                    mnemonics.append(token.text)
                    break
        return mnemonics
