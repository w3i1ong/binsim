import logging
from typing import Dict, List, TYPE_CHECKING
from binsim.utils import init_logger
from .instruction import OperandType
from binsim.disassembly.backend.binaryninja.graph import InsCFG, InsCFGNormalizer
from .datautils import ins_seq_pack_neural_input, ins_seq_collate_neural_input
from binsim.disassembly.backend.binaryninja.utils import compute_function_hash
logger = init_logger(__name__, level=logging.INFO,
                     console=True, console_level=logging.INFO)
if TYPE_CHECKING:
    from binaryninja import Function

class InsSeq(InsCFG):
    def __init__(self,
                 function_name: str,
                 adj_list: Dict[int, List],
                 features: Dict[int, List],
                 func_hash: str,
                 func_arch: str,
                 ins_num: int):
        super().__init__(function_name,
                         adj_list=adj_list,
                         features=features,
                         func_hash=func_hash,
                         func_arch=func_arch,
                         ins_num=ins_num)

    def __repr__(self):
        return f'<InsSeq::{self.name}' \
               f'(node_num={self.node_num})>'

    def as_neural_input(self, expand_time=None, max_seq_length=None):
        """
        Transform current graph to the input of neural network.
        :param expand_time: int or None
            Specify the expansion time of loop expansion.
            If None, the original graph will be returned;
            If int, its value should be a non-negative integer.
        :param max_seq_length: int or None
            Specify the maximum sequence length of the input.
            If None, the original sequence will be returned;
            If int, the sequence that exceeds the length will be truncated.
        :return:
        """
        results = []
        for line in self.features:
            results.extend(line)
            if max_seq_length is not None and len(results) >= max_seq_length:
                break
        return results[:max_seq_length]


    def as_neural_input_raw(self, max_seq_length=None, **kwargs) -> bytes:
        ins_seq = []
        for line in self.features:
            ins_seq.extend(line)
            if max_seq_length is not None and len(ins_seq) >= max_seq_length:
                break
        ins_seq = ins_seq[:max_seq_length]
        imm_values, cur_feature = [], []
        index = 0
        while index < len(ins_seq):
            operand_num = ins_seq[index]
            mnemonic = ins_seq[index + 1]
            cur_feature.extend([operand_num, mnemonic])
            index += 2
            for _ in range(operand_num):
                operand_type = ins_seq[index]
                cur_feature.append(operand_type.value)
                index += 1
                match operand_type:
                    case OperandType.REGOperand | OperandType.SpecialTokenOperand:
                        cur_feature.append(ins_seq[index])
                        index += 1
                    case OperandType.IMMOperand:
                        imm = ins_seq[index]
                        cur_feature.append(len(imm_values))
                        imm_values.append(imm)
                        index += 1
                    case OperandType.X86MemOperand:
                        base, index_reg, scale, disp = ins_seq[index:index + 4]
                        cur_feature.extend([base, index_reg, scale, len(imm_values)])
                        imm_values.append(disp)
                        index += 4
                    case OperandType.ARMMemOperand:
                        base, index_reg, shift_type, shift_value, disp = ins_seq[index:index + 5]
                        cur_feature.extend([base, index_reg, shift_type, shift_value, len(imm_values)])
                        imm_values.append(disp)
                        index += 5
                    case OperandType.ARMRegisterShiftOperand:
                        register, value, shift_type = ins_seq[index:index + 3]
                        cur_feature.extend([register, value, shift_type])
                        index += 3
                    case OperandType.ARMImmShiftOperand:
                        imm, shift_type, value = ins_seq[index:index + 3]
                        cur_feature.extend([len(imm_values), shift_type, value])
                        imm_values.append(imm)
                        index += 3
                    case OperandType.ARMVectorRegisterIndex:
                        register, index_val, vec_type = ins_seq[index:index + 3]
                        cur_feature.extend([register, index_val, vec_type])
                        index += 3
                    case OperandType.REGListOperand:
                        reg_num = ins_seq[index]
                        cur_feature.extend([reg_num] + ins_seq[index + 1:index + 1 + reg_num])
                        index += 1 + reg_num
                    case OperandType.MIPSMemOperand:
                        base, index_reg, disp = ins_seq[index:index + 3]
                        cur_feature.extend([base, index_reg, len(imm_values)])
                        imm_values.append(disp)
                        index += 3
                    case _:
                        raise ValueError(f"Meet an unknown operand type: {operand_type}.")
        return ins_seq_pack_neural_input(cur_feature, imm_values)

    @staticmethod
    def collate_raw_neural_input(inputs: List[bytes], chunks=-1, pack=False, **kwargs):
        return ins_seq_collate_neural_input(inputs, chunks, pack)


class InsSeqNormalizer(InsCFGNormalizer):
    def __init__(self, arch, token2idx: dict[str, int] | str = None):
        super().__init__(arch, token2idx)

    def __call__(self, function: "Function") -> InsSeq:
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

        cfg = InsSeq(function.name, adj_list, token_sequences,
                     func_hash=compute_function_hash(function),
                     func_arch=self.arch, ins_num=instruction_num)
        if self._ins2idx is not None:
            cfg.replace_tokens(self._ins2idx)
        return cfg
