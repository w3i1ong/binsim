import logging
from typing import List, Dict, TYPE_CHECKING
from binsim.disassembly.backend.binaryninja.utils import compute_function_hash
from binsim.disassembly.backend.binaryninja.graph.TokenCFG import TokenCFG, init_logger, TokenCFGNormalizer
from .datautils import token_seq_pack_neural_input, token_seq_collate_neural_input

if TYPE_CHECKING:
    from binaryninja import Function


logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)

class TokenSeq(TokenCFG):
    def __init__(self,
                 function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List[str]],
                 func_hash: str,
                 func_arch: str,
                 ins_num: int):
        super().__init__(function_name, adj_list=adj_list, features=features, func_hash= func_hash,
                                       func_arch=func_arch, ins_num=ins_num)

    def as_neural_input(self, max_seq_length=None,**kwargs) -> List[str]:
        """
        Convert the CFG to a sequence of tokens. Tokens are ordered by the basic block address.
        :return: A tensor of shape (token_num, ).
        """
        token_seq = []
        for basic_block in self.features:
            token_seq.extend(basic_block)
            if max_seq_length and len(token_seq) > max_seq_length:
                break
        return token_seq[:max_seq_length]

    def as_neural_input_raw(self, max_length=300, **kwargs) -> bytes:
        token_seq = self.as_neural_input(max_seq_length=max_length)
        return token_seq_pack_neural_input(token_seq, max_length)

    @staticmethod
    def collate_raw_neural_input(inputs: List[bytes], chunk_num=1, max_length=300, **kwargs):
        import torch
        if chunk_num == 1:
            chunks, lengths = token_seq_collate_neural_input(inputs, chunk_num, max_length)
            return torch.from_numpy(chunks), torch.from_numpy(lengths)
        else:
            chunks, lengths, index = token_seq_collate_neural_input(inputs, chunk_num, max_length)
            return chunks, torch.from_numpy(lengths), torch.from_numpy(index)

    def __repr__(self):
        return f'<TokenCFG::{self.name}' \
               f'(node_num={len(self._features)},arch={self.arch})>'

class TokenSeqNormalizer(TokenCFGNormalizer):
    def __call__(self, function: "Function") -> TokenSeq:
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
        cfg = TokenSeq(function.name, adj_list, token_sequences,
                       func_hash=compute_function_hash(function),
                       func_arch=self.arch, ins_num=instruction_num)
        if self._ins2idx is not None:
            cfg.replace_tokens(self._ins2idx)
        return cfg
