from abc import ABC
import numpy as np
import torch
from binsim.disassembly.backend.binaryninja import BSInsOperandType, BSInstruction
from binsim.utils import diagonalize_matrices
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union
import random
from collections import OrderedDict
from mknapsack import solve_bin_packing


class InstructionsSeqBase(ABC):
    def __init__(self):
        self._regs = None
        self._imm = None
        self._mnemic = None
        self._mem_op_idx = None
        self._operand_idx = None
        self._arch = None
        self._mnemic_idx = None

    def to(self, device):
        self._regs = self._regs.to(device)
        self._imm = self._imm.to(device)
        self._mnemic = self._mnemic.to(device)
        self._mem_op_idx = tuple(map(lambda x: x.to(device), self._mem_op_idx))
        self._operand_idx = self.operand_idx.to(device)
        self._mnemic_idx = self._mnemic_idx.to(device)
        return self

    @property
    def mnemic_idx(self) -> torch.Tensor:
        return self._mnemic_idx

    @property
    def arch(self):
        return self._arch

    @property
    def mnemic(self) -> torch.Tensor:
        return self._mnemic

    @property
    def regs(self) -> torch.Tensor:
        return self._regs

    @property
    def imm(self) -> torch.Tensor:
        return self._imm

    @property
    def mem_op_idx(self) -> torch.Tensor:
        return self._mem_op_idx

    @property
    def operand_idx(self) -> torch.Tensor:
        return self._operand_idx


class BatchedInstructionSeq(InstructionsSeqBase):
    MEMORY_OP_NUM = {
        "mips": 3,
        "arm": 5,
        "x86": 4
    }
    """
    This class is used to store a batch of instruction sequences. The batched instruction sequences can be fixed length
    or variable length. The variable length instruction sequences are padded with a random instruction.
    """

    def __init__(self, arch, ins_seq=None):
        super().__init__()
        self._arch = arch
        if ins_seq is not None:
            self._init_tokens(ins_seq)

    def _init_tokens(self, ins_seq, *args, **kwargs):
        self._mnemic, self._mnemic_idx, self._regs, self._imm, self._operand_idx, self._mem_op_idx = \
            self._real_init_tokens(ins_seq)

    def _real_init_tokens(self, ins_sequence_list: List[List[BSInstruction]], need_verify=False):
        """
        Initialize this instruction sequence with a list of instruction sequences.
        :param ins_sequence_list: a list of instruction sequences.
        :return:
        """
        mnemic_id2idx = defaultdict(lambda: len(mnemic_id2idx))
        regs_id2idx = defaultdict(lambda: len(regs_id2idx))
        imm_id2idx = defaultdict(lambda: len(imm_id2idx))
        mem_id2idx = defaultdict(lambda: len(mem_id2idx))
        mnemic_idx_seq, operand_idx_seq = [], []
        max_seq_length = max(map(lambda x: len(x), ins_sequence_list)) if len(ins_sequence_list) > 0 else 0
        # do some initialization operations:
        #   1. initialize mnemic index sequence
        #   2. collect all register tokens and immediate tokens.
        for instructions in ins_sequence_list:
            mnemic_idx = []
            for ins_idx, ins in enumerate(instructions):
                mnemic_idx.append(mnemic_id2idx[ins.mnemonic])
                for op in ins.operands:
                    if op.type == BSInsOperandType.REG:
                        _ = regs_id2idx[op.value]
                    elif op.type == BSInsOperandType.IMM:
                        _ = imm_id2idx[op.value]
                    elif op.type in [BSInsOperandType.MIPS_MEM, BSInsOperandType.X86_MEM, BSInsOperandType.ARM_MEM]:
                        _ = mem_id2idx[op.value]
                        if self.arch == 'arm':
                            base, index, shift, disp, flag = op.value
                            _ = regs_id2idx[base], regs_id2idx[index], imm_id2idx[shift], imm_id2idx[disp], regs_id2idx[
                                flag]
                        elif self.arch == 'x86':
                            base, index, scale, disp = op.value
                            _ = regs_id2idx[base], regs_id2idx[index], imm_id2idx[scale], imm_id2idx[disp]
                        elif self.arch == 'mips':
                            base, index, disp = op.value
                            _ = regs_id2idx[base], regs_id2idx[index], imm_id2idx[disp]
                    else:
                        raise NotImplementedError("Unknown operand type: {}".format(op.type))
            # pad with a random instruction if the sequence is shorter than the max length.
            mnemic_idx.extend([0] * (max_seq_length - len(mnemic_idx)))
            mnemic_idx_seq.append(mnemic_idx)
        # initialize mnemic index sequence
        _mnemic_idx = torch.from_numpy(np.array(mnemic_idx_seq, dtype=np.int64))
        # initialize operand index, register index, immediate index, memory index.
        # after we generate the embedding for each operand, we will use operand index to generate the operand embedding
        # sequence for each instruction. Just like:
        #  reg_embedding = ...
        #  imm_embedding = ...
        #  mem_embedding = ...
        #  operand_embedding = torch.cat([reg_embedding, imm_embedding, mem_embedding], dim=0)[ins_operand_idx_seq]
        imm_op_idx_base = len(regs_id2idx)
        mem_op_idx_base = imm_op_idx_base + len(imm_id2idx)
        for instructions in ins_sequence_list:
            function_operand_idx_seq = []
            for ins in instructions:
                ins_operand_idx_seq = []
                for op in ins.operands:
                    if op.type == BSInsOperandType.REG:
                        ins_operand_idx_seq.append(regs_id2idx[op.value])
                    elif op.type == BSInsOperandType.IMM:
                        ins_operand_idx_seq.append(imm_op_idx_base + imm_id2idx[op.value])
                    elif op.type in [BSInsOperandType.MIPS_MEM, BSInsOperandType.X86_MEM, BSInsOperandType.ARM_MEM]:
                        ins_operand_idx_seq.append(mem_op_idx_base + mem_id2idx[op.value])

                function_operand_idx_seq.append(ins_operand_idx_seq)
            # if the instruction sequence is shorter than the max length, we pad its operand index sequence with zeros.
            function_operand_idx_seq.extend(
                [[0] * len(function_operand_idx_seq[0])] * (max_seq_length - len(function_operand_idx_seq)))
            operand_idx_seq.append(function_operand_idx_seq)
        # initialize token id sequence
        _mnemic = self._get_id_seq(mnemic_id2idx)
        _regs = self._get_id_seq(regs_id2idx)
        _imm = self._get_id_seq(imm_id2idx)
        # initialize token index sequence
        _operand_idx = torch.from_numpy(np.array(operand_idx_seq, dtype=np.int64))
        # initialize token index for memory operands
        mem_op_idx = [[0 for _ in range(len(mem_id2idx))] for _ in range(self.MEMORY_OP_NUM[self.arch])]
        for mem_op, idx in mem_id2idx.items():
            for op_idx, value_map in zip(range(len(mem_op)),
                                         [regs_id2idx, regs_id2idx, imm_id2idx, imm_id2idx, regs_id2idx]):
                mem_op_idx[op_idx][idx] = value_map[mem_op[op_idx]]
        _mem_op_idx = [torch.from_numpy(np.array(op, dtype=np.int64)) for op in mem_op_idx]
        _mnemic = torch.tensor(_mnemic)
        _regs = torch.tensor(_regs)
        _imm = torch.unsqueeze(torch.tensor(_imm, dtype=torch.float32), dim=1)
        if need_verify:
            self.verify_mnemic(ins_sequence_list, _mnemic, _mnemic_idx)
            self.verify_operands(ins_sequence_list, _regs, _imm, _mem_op_idx, _operand_idx)
        return _mnemic, _mnemic_idx, _regs, _imm, _operand_idx, _mem_op_idx

    @staticmethod
    def verify_mnemic(ins_sequence_list: List[List[BSInstruction]], mnemic, mnemic_idx):
        # verify the mnemic index sequence
        mnemic = mnemic[mnemic_idx]
        for mnemic_seq, function in zip(mnemic, ins_sequence_list):
            for mnemic_id, ins in zip(mnemic_seq, function):
                assert mnemic_id == ins.mnemonic

    def verify_operands(self,
                        ins_sequence_list: List[List[BSInstruction]],
                        regs: torch.Tensor,
                        imm: torch.Tensor,
                        mem_op_idx: List,
                        operand_idx_seq: torch.Tensor):
        operand_list: List[np.array] = list()
        # add register operands
        operand_list.extend(regs.numpy())
        # add immediate operands
        operand_list.extend(imm.numpy())
        # add memory operands
        for ops in zip(*mem_op_idx):
            if self.arch == 'x86':
                base, index, scale, offset = ops
                operand_list.append((regs[base], regs[index], imm[scale], imm[offset]))
            elif self.arch == 'arm':
                base, index, shift, offset, flag = ops
                operand_list.append((regs[base], regs[index], imm[shift], imm[offset], regs[flag]))
            elif self.arch == 'mips':
                base, index, offset = ops
                operand_list.append((regs[base], regs[index], imm[offset]))
            else:
                raise ValueError("Unsupported architecture: {}".format(self.arch))
        # verify the operand index sequence
        for op_seq, ins_seq in zip(operand_idx_seq, ins_sequence_list):
            for op_idx, ins in zip(op_seq, ins_seq):
                for op_id, op in zip(op_idx, ins.operands):
                    if op.type == BSInsOperandType.REG:
                        assert operand_list[op_id].item() == op.value
                    elif op.type == BSInsOperandType.IMM:
                        if abs(op.value) > 100:
                            assert (operand_list[op_id].item() - op.value) / op.value < 1e-6
                        else:
                            assert operand_list[op_id].item() == op.value
                    else:
                        assert operand_list[op_id] == op.value

    @staticmethod
    def _get_id_seq(token_id2idx: Dict[Any, int]) -> List[Any]:
        res = [None for _ in range(len(token_id2idx))]
        for token_id, idx in token_id2idx.items():
            res[idx] = token_id
        return res


class BatchedFunctionSeq(BatchedInstructionSeq):
    def __init__(self,
                 arch: str,
                 ins_seq: List[List[BSInstruction]],
                 relative_position_offset: List[torch.Tensor],
                 masked_ins_idx: List[List[int]],
                 masked_ins: List[List[BSInstruction]],
                 predict_classes=20):
        """
        Batched function sequence for masked language model.
        :param arch: the architecture of the instruction sequence.
        :param ins_seq: the instruction sequence.
        :param relative_position_offset: the relative position offset between each instruction pair.
        :param masked_ins_idx: the index of instructions that are replaced by mask instruction.
        :param masked_ins: the original instruction that are replaced by mask instruction.
        :param predict_classes: the number of classes to predict, we will randomly select another predict_classes-1
            instructions and let the model identify the original instruction from these predict_classes instructions.
        """
        super().__init__(arch)
        # data for training
        self._masked_predict = None
        self._masked_predict_idx = None

        self._arch = arch
        self._predict_classes = predict_classes

        ins_seq, relative_position_offset, masked_ins_idx, masked_ins = \
            self._merge_short_sequence(ins_seq=ins_seq,
                                       relative_position_offset=relative_position_offset,
                                       masked_ins_idx=masked_ins_idx,
                                       masked_ins=masked_ins)
        self._init_tokens(ins_seq, masked_ins_idx, masked_ins)

        self._relative_position_offset = relative_position_offset
        self._ins_seq = ins_seq

    def to(self, device):
        self._masked_predict = self._masked_predict.to(device)
        self._masked_predict_idx = self._masked_predict_idx.to(device)
        self._relative_position_offset = self._relative_position_offset.to(device)
        super().to(device)
        return self

    @property
    def masked_predict(self):
        return self._masked_predict

    @property
    def masked_predict_idx(self):
        return self._masked_predict_idx

    @property
    def relative_position_offset(self):
        return self._relative_position_offset

    @staticmethod
    def _merge_short_sequence(ins_seq: List[List],
                              relative_position_offset: List[torch.Tensor],
                              masked_ins_idx: List[List[int]],
                              masked_ins: List[List[BSInstruction]]):
        # use bin packing to merge short sequences int a long sequence
        seq_len = list(map(len, ins_seq))
        max_len = max(seq_len)
        bin_loc = solve_bin_packing(seq_len, max_len)
        # number of sequence
        # bin_num = max(bin_loc)
        # which long sequence does each short sequence belong to
        bin_loc = [loc - 1 for loc in bin_loc]
        bin_items = OrderedDict()
        for seq_id, bin_id in enumerate(bin_loc):
            if bin_id not in bin_items:
                bin_items[bin_id] = []
            bin_items[bin_id].append(seq_id)

        # merge sequences according to bin_loc
        merged_ins_seq, merged_relative_position = [], []
        merged_masked_ins_idx, merged_masked_ins = [], []

        # merge short sequences
        for bin_id, seq_id_list in bin_items.items():
            cur_seq = []
            for seq_id in seq_id_list:
                cur_seq.extend(ins_seq[seq_id])
            merged_ins_seq.append(cur_seq)

        # merge relative position offset
        for bin_id, seq_id_list in bin_items.items():
            cur_relative_position = []
            base = 0
            for seq_id in seq_id_list:
                cur_relative_position.append(relative_position_offset[seq_id])
                base += len(ins_seq[seq_id])

            merged_relative_position.append(diagonalize_matrices(cur_relative_position, col=max_len, row=max_len))
        merged_relative_position = torch.stack(merged_relative_position)

        # merge masked ins idx
        for bin_id, seq_id_list in bin_items.items():
            cur_masked_ins_idx = []
            base = 0
            for seq_id in seq_id_list:
                cur_masked_ins_idx.extend([idx + base for idx in masked_ins_idx[seq_id]])
                base += len(ins_seq[seq_id])
            merged_masked_ins_idx.append(cur_masked_ins_idx)

        # merge masked ins
        for bin_id, seq_id_list in bin_items.items():
            cur_masked_ins = []
            for seq_id in seq_id_list:
                cur_masked_ins.extend(masked_ins[seq_id])
            merged_masked_ins.append(cur_masked_ins)

        return merged_ins_seq, merged_relative_position, merged_masked_ins_idx, merged_masked_ins

    def _init_tokens(self, ins_sequence_list: List[List[BSInstruction]], masked_ins_idx, masked_ins, *args,
                     **kwargs):
        # replace those replaced instructions with random-selected instructions
        all_instructions = [ins for ins_sequence in ins_sequence_list for ins in ins_sequence if ins is not None]
        for ins_seq in ins_sequence_list:
            for ins_idx, ins in enumerate(ins_seq):
                if ins is None:
                    ins_seq[ins_idx] = random.choice(all_instructions)
        # randomly select self._predict_classes-1 candidates for each masked instruction, including the original instruction.
        candidate_ins = []
        for function_masked_ins in masked_ins:
            for ins in function_masked_ins:
                candidate_ins.append([ins] + random.choices(all_instructions, k=self._predict_classes - 1))
        # initialize instruction sequence
        self._mnemic, self._mnemic_idx, self._regs, self._imm, self._operand_idx, self._mem_op_idx = \
            self._real_init_tokens(ins_sequence_list)
        # initialize masked instruction index, which is used to calculate the loss.
        # to make the calculation easier, we flatten the instruction sequence.
        # The shape of the output of the model is (batch_size, max_seq_len, vocab_size).
        # To get the embedding of masked instructions, we need to flatten the sequence.
        # That's to say, reshape the output to (batch_size * max_seq_len, vocab_size).
        # Then we can get the embedding of masked instructions by indexing the output with masked instruction index.
        max_seq_len = max(map(lambda x: len(x), ins_sequence_list))
        assert max_seq_len == self._mnemic_idx.size(-1)
        masked_ins_idx = [ins_idx + row_idx * max_seq_len for row_idx, ins_idx_seq in enumerate(masked_ins_idx) for
                          ins_idx in ins_idx_seq]

        self._masked_predict = BatchedInstructionSeq(self.arch, candidate_ins)
        self._masked_predict_idx = torch.tensor(masked_ins_idx, dtype=torch.long)
