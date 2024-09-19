import os
import dgl
import torch
import random
import logging
import numpy as np
from ...base import CFGBase, CFGNormalizerBase
from enum import Enum
from dgl import DGLGraph
from typing import Any
from typing import Dict, List, Set, Tuple, Union
from binsim.utils import fastGraph
from binaryninja import Function, BasicBlock, InstructionTextTokenType as TokenType
from ..utils import compute_function_hash
from itertools import chain
from binsim.utils import init_logger

logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)


class TokenCFGDataForm(Enum):
    InsStrSeq = 'InsStrSeq'
    InsStrGraph = 'InsStrGraph'
    TokenSeq = 'TokenSeq'
    TokenGraph = 'TokenGraph'


class TokenCFGMode(Enum):
    ID = 'ID'
    TOKEN = 'TOKEN'


class TokenCFG(CFGBase):
    def __init__(self,
                 function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List[List[str]]],
                 func_hash: str,
                 func_arch: str,
                 node_num: int,
                 form=TokenCFGDataForm.InsStrGraph,
                 entry_points=None):
        """
        Build a token CFG from the given adjacency list and features.
        :param function_name: The name of the function.
        :param adj_list: The adjacency list of the CFG.
        :param features: A list, in which each element is a list of tokens.
        :param func_hash: The hash of the function, used to remove duplicate functions.
        :param func_arch: The architecture of the function.
        :param form: The form of the input.
            If set to 'sequence', a sequence of tokens will be returned.
            If set to 'graph', a DGLGraph and two tensors will be returned. The first tensor is a block_num * max_token_num
            tensor, which represents the instruction sequences in each basic block. The second tensor is a block_num tensor,
            which represents the length of instruction sequence in each basic block.
        """
        super(TokenCFG, self).__init__(function_name, func_hash=func_hash, func_arch=func_arch,
                                       entries=entry_points, node_num=node_num)
        self._adj_list = adj_list
        self._features: Dict[int, List[Union[List[str], int, str]]] = features
        self._mode = TokenCFGMode.TOKEN
        self._form = form

    @property
    def data_form(self):
        return self._form

    @data_form.setter
    def data_form(self, form: TokenCFGDataForm):
        self._form = form

    def unique_tokens(self) -> Set[Union[str, int]]:
        """
        Get the all unique tokens in the CFG.
        :return: A set of unique tokens.
        """
        result = set()
        for basic_block in self.features.values():
            for instruction in basic_block:
                result.update(instruction)
        return result

    def replace_tokens(self, token2id: Dict[str, int],
                       record_unseen=False,
                       update_token2id=False) -> Union[Set, int, int] | None:
        """
        Replace tokens in the CFG with token ids.
        :param token2id: A dictionary, which maps tokens to token ids.
        :param record_unseen: Whether to record unseen tokens.
        :return:
        """
        if self._mode == TokenCFGMode.ID:
            return None
        unseen_tokens = set()
        total_token_num, unknown_token_num = 0, 0
        for address, token_list in self.features.items():
            if self.data_form in [TokenCFGDataForm.InsStrGraph, TokenCFGDataForm.InsStrSeq]:
                tokens = [''.join(token) for token in token_list]
            else:
                tokens = list(chain.from_iterable(token_list))
            total_token_num += len(tokens)

            for idx, token in enumerate(tokens):
                token_id = token2id.get(token, 0)
                if token_id == 0:
                    unknown_token_num += 1
                    if record_unseen:
                        unseen_tokens.add(token)
                    if update_token2id:
                        token_id = len(token2id)
                        token2id[token] = token_id
                tokens[idx] = token_id
            self._features[address] = tokens
        self._mode = TokenCFGMode.ID
        return unseen_tokens, total_token_num, unknown_token_num

    def as_neural_input(self, expand_time=None, max_seq_length=None) -> Union[Tuple[DGLGraph, List, List], List]:
        """
        Convert the CFG to suitable input for neural network.
        :return:
        """
        assert self._mode == TokenCFGMode.ID, f"The CFG is now in {self._mode} mode. " \
                                              f"You should call replace_tokens() to convert tokens to token ids first."
        if self.data_form in [TokenCFGDataForm.InsStrGraph, TokenCFGDataForm.TokenGraph]:
            return self._as_graph(expand_time=expand_time, max_seq_length=max_seq_length)
        elif self.data_form in [TokenCFGDataForm.InsStrSeq, TokenCFGDataForm.TokenSeq]:
            return self._as_sequence(max_seq_length=max_seq_length)
        else:
            raise NotImplementedError()

    def _as_sequence(self, max_seq_length=None) -> List:
        """
        Convert the CFG to a sequence of tokens. Tokens are ordered by the basic block address.
        :return: A tensor of shape (token_num, ).
        """
        token_seq = []
        for address, basic_block in self.features.items():
            token_seq.extend(basic_block)
            if max_seq_length and len(token_seq) > max_seq_length:
                break
        return token_seq[:max_seq_length]

    def _as_graph(self, expand_time=None, max_seq_length=None) -> Tuple[DGLGraph, List, List]:
        """
        Convert the CFG to a DGLGraph and two tensors.
        The first tensor is a block_num * max_token_num tensor, which represents the instruction sequences in each basic
        block.
        The second tensor is a block_num tensor, which represents
        the length of instruction sequence in each basic block.

        :return:
        """
        cfg, id2addr = self.as_dgl_graph(expand_time=expand_time)
        token_seq_list, lengths = [None]*len(id2addr), [None]*len(id2addr)
        for idx, addr in id2addr.items():
            if max_seq_length is None:
                token_seq_list[idx] = self.features[addr][:]
                lengths[idx] = len(self.features[addr])
            else:
                token_seq_list[idx] = self.features[addr][:max_seq_length]
                lengths[idx] = min(len(self.features[addr]), max_seq_length)
        return cfg, token_seq_list, lengths

    def as_token_list(self) -> List[str]:
        """
        Get all tokens in the CFG.
        :return: A list of tokens.
        """
        results = []
        if self.data_form in [TokenCFGDataForm.InsStrGraph, TokenCFGDataForm.InsStrSeq]:
            for addr in self.basic_block_address:
                results.extend([str(instruction) for instruction in self._features[addr]])
        else:
            for addr in self.basic_block_address:
                for instruction in self._features[addr]:
                    results.extend(list(map(str, instruction)))
        return results

    def random_walk(self, walk_num=10, max_walk_length=100,
                    basic_block_func=None,
                    edge_coverage=True) -> List[List[str]]:
        """
        Random walk on the CFG.
        :param walk_num: How many random walks to perform.
        :param max_walk_length: The maximum length of a walk.
        :param basic_block_func: A function that converts basic block to a Instruction.
        :param edge_coverage: Whether to use edge coverage to ensure that all edges are covered.
        :return: A list of tokens.
        """
        results = []
        cached_edges = {}

        if basic_block_func is not None:
            for addr, basic_block in self._features.items():
                cached_edges[addr] = basic_block_func(basic_block)
            for _ in range(walk_num):
                walk_path, addr = [], random.choice(self.entries)
                while True:
                    walk_path.extend(cached_edges[addr])
                    # update the max walk length and check whether the path length exceeds the limit
                    max_walk_length -= len(self._features[addr])
                    if max_walk_length <= 0 or len(self.adj_list[addr]) == 0:
                        break
                    # randomly choose a successor
                    addr = random.choice(self._adj_list[addr])
                results.append(walk_path)

            if edge_coverage:
                # ensure that all edges are covered
                for addr in self.basic_block_address:
                    for next_addr in self.adj_list[addr]:
                        results.append(cached_edges[addr] + cached_edges[next_addr])
        else:
            ins_as_word = (self.data_form in [TokenCFGDataForm.InsStrGraph, TokenCFGDataForm.InsStrSeq])
            for _ in range(walk_num):
                walk_path, addr = [], random.choice(self.entries)
                while True:
                    if ins_as_word:
                        walk_path.extend([''.join(ins) for ins in self._features[addr]])
                    else:
                        for ins in self._features[addr]:
                            walk_path.extend(ins)
                    # update the max walk length and check whether the path length exceeds the limit
                    max_walk_length -= len(self._features[addr])
                    if max_walk_length <= 0 or len(self.adj_list[addr]) == 0:
                        break
                    # randomly choose a successor
                    addr = random.choice(self._adj_list[addr])
                results.append(walk_path)

            if edge_coverage:
                # ensure that all edges are covered
                for addr in self.basic_block_address:
                    for next_addr in self.adj_list[addr]:
                        if ins_as_word:
                            results.append([''.join(ins) for ins in self._features[addr]] +
                                           [''.join(ins) for ins in self._features[next_addr]])
                        else:
                            for ins in self._features[addr]:
                                results.extend(ins)
                            for ins in self._features[next_addr]:
                                results.extend(ins)
        return results

    def __repr__(self):
        return f'<TokenCFG::{self.name}' \
               f'(node_num={len(self._features)},arch={self.arch.value})>'

    def isReducible(self):
        u, v = [], []
        addr2idx = dict(zip(self.basic_block_address, range(len(self.basic_block_address))))
        for start, ends in self.adj_list.items():
            u.extend([addr2idx[start]] * len(ends))
            v.extend([addr2idx[end] for end in ends])

        graph = fastGraph(len(self.features), u, v, 0)
        acyclic, reducible = graph.isAcyclic(), graph.isReducible()
        return reducible


class TokenCFGNormalizer(CFGNormalizerBase):
    def __init__(self, arch, imm_threshold=None, offset_threshold=None, ins2idx: dict[str, int] = None):
        super(TokenCFGNormalizer, self).__init__(arch=arch)
        self._imm_limit = imm_threshold
        self._offset_limit = offset_threshold
        self._ins2idx = ins2idx
        self._call_instructions = self.get_call_instructions()

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

    def __call__(self, function: Function) -> TokenCFG:
        adj_list, entry_points = self.extract_adj_list(function)
        token_sequences = self._extract_tokens(function)
        # some basic blocks contain no instruction and no successors, just remove them
        for basic_block in function.basic_blocks:
            if basic_block.instruction_count == 0 and len(basic_block.outgoing_edges) == 0:
                token_sequences.pop(basic_block.start)
                adj_list.pop(basic_block.start)
                for edge in basic_block.incoming_edges:
                    src_addr = edge.source.start
                    adj_list[src_addr].remove(basic_block.start)
        # end of the check
        cfg = TokenCFG(function.name, adj_list, token_sequences,
                       func_hash=compute_function_hash(function),
                       func_arch=self.arch, node_num=len(adj_list))
        cfg.data_form = TokenCFGDataForm.InsStrGraph
        if self._ins2idx is not None:
            cfg.replace_tokens(self._ins2idx)
        return cfg

    def _extract_tokens(self, function: Function) -> Dict[int, np.array]:
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

    def extract_bb_tokens(self, basic_block: BasicBlock) -> List[List[str]]:
        instructions = []
        for instruction, ins_size in basic_block:
            instructions.append(self._parse_instruction(instruction))
        return instructions

    def _parse_instruction(self, instruction) -> List[str]:

        results = []
        idx = 0

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
