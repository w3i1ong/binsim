import os
import logging
import numpy as np
from enum import Enum
from binsim.utils import init_logger
from typing import Dict, List, Set, Union, TYPE_CHECKING
from binsim.disassembly.backend.binaryninja.utils import compute_function_hash
from binsim.disassembly.core import BinsimCFG, CFGNormalizerBase
from .datautils import token_cfg_pack_neural_input, token_cfg_collate_neural_input

logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)

if TYPE_CHECKING:
    from binaryninja import Function, BasicBlock

class TokenCFGMode(Enum):
    ID = 'ID'
    TOKEN = 'TOKEN'


class TokenCFG(BinsimCFG):
    def __init__(self,
                 function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List[str]],
                 func_hash: str,
                 func_arch: str,
                 ins_num: int):
        """
        Build a token CFG from the given adjacency list and features.
        :param function_name: The name of the function.
        :param adj_list: The adjacency list of the CFG.
        :param features: A list, in which each element is a list of tokens.
        :param func_hash: The hash of the function, used to remove duplicate functions.
        :param func_arch: The architecture of the function.
        """
        super(TokenCFG, self).__init__(function_name, func_hash=func_hash, func_arch=func_arch,
                                       features=features, adj_list=adj_list, ins_num=ins_num)
        self._mode = TokenCFGMode.TOKEN

    def unique_tokens(self) -> Set[Union[str, int]]:
        """
        Get the all unique tokens in the CFG.
        :return: A set of unique tokens.
        """
        result = set()
        for basic_block in self.features:
            result.update(basic_block)
        return result

    def replace_tokens(self, token2id: Dict[str, int]):
        """
        Replace tokens in the CFG with token ids.
        :param token2id: A dictionary, which maps tokens to token ids.
        :return:
        """
        if self._mode == TokenCFGMode.ID:
            return None
        for idx, tokens in enumerate(self.features):
            self._features[idx] = [token2id.get(token, 0) for token in tokens]
        self._mode = TokenCFGMode.ID

    def as_neural_input(self, expand_time=None, max_seq_length=None):
        """
        Convert the CFG to a DGLGraph and two tensors.
        The first tensor is a block_num * max_token_num tensor, which represents the instruction sequences in each basic
        block.
        The second tensor is a block_num tensor, which represents
        the length of instruction sequence in each basic block.

        :return:
        """
        import dgl
        src, dst, node_num, nodeId = self.prepare_graph_info(expand_time=expand_time)
        token_seq_list, lengths = [None]*len(self.features), [None]*len(self.features)
        for idx, bb_features in enumerate(self.features):
            if max_seq_length is None:
                token_seq_list[idx] = bb_features[:]
                lengths[idx] = len(bb_features)
            else:
                token_seq_list[idx] = bb_features[:max_seq_length]
                lengths[idx] = min(len(bb_features), max_seq_length)
        cfg: dgl.DGLGraph = dgl.graph((src, dst), num_nodes=node_num)
        return cfg, token_seq_list, lengths

    def as_neural_input_raw(self, max_length=300, **kwargs):
        src, dst, node_num, nodeId = self.prepare_graph_info(expand_time=None)
        features = [bb[:max_length] for bb in self.features]
        return token_cfg_pack_neural_input(node_num, src, dst, features)


    @staticmethod
    def collate_raw_neural_input(inputs: List[bytes], chunks=1, **kwargs):
        import torch, dgl
        if chunks <= 1:
            edges, (features, lengths, node_nums) = token_cfg_collate_neural_input(inputs, chunks)
            graphs = []
            for (src, dst), node_num in zip(edges, node_nums):
                graphs.append(dgl.graph((src.astype(np.int32), dst.astype(np.int32)), num_nodes=node_num))
            graph = dgl.batch(graphs)
            return graph, torch.from_numpy(features), torch.from_numpy(lengths)
        else:
            edges, (lengths, index, node_nums, *chunks) = token_cfg_collate_neural_input(inputs, chunks)
            graphs = []
            for (src, dst), node_num in zip(edges, node_nums):
                graphs.append(dgl.graph((src.astype(np.int32), dst.astype(np.int32)), num_nodes=node_num))
            graph = dgl.batch(graphs)
            chunks = [torch.from_numpy(chunk) for chunk in chunks]
            return graph, chunks, torch.from_numpy(lengths), torch.from_numpy(index)

    def as_token_list(self) -> List[str]:
        """
        Get all tokens in the CFG.
        :return: A list of tokens.
        """
        results = []
        for line in self.features:
            results.extend(line)
        return results

    def __repr__(self):
        return f'<TokenCFG::{self.name}' \
               f'(node_num={len(self._features)},arch={self.arch})>'

    def minimize(self):
        self._mode = None
        super().minimize()
        return self


class TokenCFGNormalizer(CFGNormalizerBase):
    def __init__(self, arch, imm_threshold=None, offset_threshold=None, ins_as_token = True,
                 ins2idx: dict[str, int] = None):
        super(TokenCFGNormalizer, self).__init__(arch=arch)
        self._imm_limit = imm_threshold
        self._offset_limit = offset_threshold
        self._ins2idx = ins2idx
        self._call_instructions = self.get_call_instructions()
        self._ins_as_token = ins_as_token

    @staticmethod
    def get_call_instructions():
        results = []
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        mnem_path = os.path.join(cur_dir, 'data', 'mnem')
        for arch in os.listdir(mnem_path):
            call_path = os.path.join(mnem_path, arch, 'call')
            if not os.path.exists(call_path):
                continue
            with open(call_path, 'r') as f:
                results.extend([line.strip() for line in f.readlines()])
        return set(results)

    def __call__(self, function: "Function") -> TokenCFG:
        adj_list, entry_points = self.extract_adj_list(function)
        token_sequences = self._extract_tokens(function)
        instruction_num = 0
        # some basic blocks contain no instruction and no successors, just remove them
        for basic_block in function.basic_blocks:
            if basic_block.instruction_count == 0 and len(basic_block.outgoing_edges) == 0:
                token_sequences.pop(basic_block.start)
                adj_list.pop(basic_block.start)
                for edge in basic_block.incoming_edges:
                    src_addr = edge.source.start
                    adj_list[src_addr].remove(basic_block.start)
            instruction_num += basic_block.instruction_count
        # end of the check
        cfg = TokenCFG(function.name, adj_list, token_sequences,
                       func_hash=compute_function_hash(function),
                       func_arch=self.arch, ins_num=instruction_num)
        if self._ins2idx is not None:
            cfg.replace_tokens(self._ins2idx)
        return cfg

    def _extract_tokens(self, function: "Function") -> Dict[int, List[str]]:
        bb_tokens = {}
        for basic_block in function.basic_blocks:
            bb_tokens[basic_block.start] = self.extract_bb_tokens(basic_block)
        return bb_tokens

    def _get_imm_tokens(self, imm) -> str:
        if self._imm_limit is not None and abs(imm) >= self._imm_limit:
            return '[IMM]'
        if imm >= 0:
            return f'{hex(imm)}'
        else:
            return f'-{hex(-imm)}'

    def _get_offset_tokens(self, offset) -> str:
        if self._offset_limit is not None and abs(offset) >= self._offset_limit:
            return '[OFFSET]'

        if offset >= 0:
            return f'+{hex(offset)}'
        elif offset < 0:
            return f'-{hex(-offset)}'

    def extract_bb_tokens(self, basic_block: "BasicBlock") -> List[str]:
        instructions = []
        for instruction, ins_size in basic_block:
            ins_tokens = self._parse_instruction(instruction)
            if self._ins_as_token:
                instructions.append('_'.join(ins_tokens))
            else:
                instructions.extend(ins_tokens)
        return instructions

    def _parse_instruction(self, instruction) -> List[str]:
        from binaryninja import InstructionTextTokenType as TokenType
        results,idx = [], 0
        if instruction[0].type == TokenType.InstructionToken:
            if instruction[0].text in self._call_instructions:
                results.append(instruction[0].text)
                target = instruction[1]
                if target.type != TokenType.RegisterToken:
                    results.append("[Func]")
                else:
                    results.append(target.text)
                return results

        while idx < len(instruction):
            token = instruction[idx]
            if token.type == TokenType.TextToken and token.text.strip() == '':
                idx += 1
                continue

            match token.type:
                case TokenType.IntegerToken:
                    results.append(self._get_imm_tokens(token.value))
                case TokenType.PossibleAddressToken:
                    results.append(self._get_offset_tokens(token.value))
                case TokenType.FloatingPointToken:
                    results.append(self._get_imm_tokens(token.value))
                case TokenType.CodeRelativeAddressToken:
                    results.append(self._get_offset_tokens(token.value))
                case TokenType.BeginMemoryOperandToken:
                    first_token_text, second_token_text = None, None
                    if idx + 1 < len(instruction):
                        first_token_text = instruction[idx+1].text.strip()
                    if idx + 2 < len(instruction):
                        second_token_text = instruction[idx+2].text.strip()
                    if first_token_text == 'rel' or second_token_text == 'rel':
                        results.append('[MEM]')
                        while idx < len(instruction) and instruction[idx].type != TokenType.EndMemoryOperandToken:
                            idx += 1
                    else:
                        results.append(token.text.strip())
                        idx += 1
                        while idx < len(instruction) and instruction[idx].type != TokenType.EndMemoryOperandToken:
                            token = instruction[idx]
                            if token.type == TokenType.TextToken and token.text.strip() == '':
                                idx += 1
                                continue
                            elif token.type == TokenType.IntegerToken:
                                results.append(self._get_imm_tokens(token.value))
                            else:
                                results.append(token.text.strip())
                            idx += 1
                        if idx < len(instruction):
                            results.append(instruction[idx].text.strip())
                case _:
                    results.append(token.text.strip())

            idx += 1
        return results
