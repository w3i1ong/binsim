from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from binsim.disassembly.backend.binaryninja import ProgramDependencyGraph
from .instruction import BatchedFunctionSeq
from binsim.disassembly.backend.binaryninja import BSInstruction, BSInsOperandType
import torch
import random


class InstructionDataset(Dataset):
    def __init__(self,
                 token2id: Dict[str, int],
                 data: List[ProgramDependencyGraph],
                 candidate_instructions: int = 20,
                 with_control_flow: bool = True):
        """
        :param token2id: a dict mapping token to id.
        :param data: a list of ProgramDependencyGraph.
        :param candidate_instructions: the number of candidate instructions for each masked instruction.
        :param with_control_flow: whether to use control flow mask.
        """
        super().__init__()
        self._data = data
        self._token2id = token2id
        self._predict_classes = candidate_instructions
        self._with_control_flow = with_control_flow

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        # replace tokens with ids
        self._data[item].replace_tokens(self._token2id)
        arch = self._data[item].arch
        ins_seq, control_flow_relative_offset = self._data[item].as_neural_input(
            with_control_flow=self._with_control_flow)
        # as we will modify the ins_seq, we need to copy it
        ins_seq = ins_seq[:]
        masked_ins_idx, masked_ins = [], []
        # randomly replace instructions with <unk> token
        mask_mnemic = self._token2id['<NO_MNEM>']
        mask_op = self._token2id['<NO_REG>']
        mask_op = BSInsOperand(BSInsOperandType.REG, mask_op)
        # todo: different architecture has different number of registers.
        mask_ins = BSInstruction(mask_mnemic, [mask_op, mask_op, mask_op, mask_op])
        for i in range(len(ins_seq)):
            prob = random.random()
            if prob < 0.15:
                masked_ins.append(ins_seq[i])
                prob /= 0.15
                if prob < 0.8:
                    ins_seq[i] = mask_ins
                elif prob < 0.9:
                    ins_seq[i] = None
                masked_ins_idx.append(i)
        return arch, (ins_seq, control_flow_relative_offset, masked_ins_idx, masked_ins)

    @staticmethod
    def collate_fn(data: List[Tuple[str, Tuple[List[BSInstruction], List[List[int]], torch.Tensor]]]):
        arch, data = zip(*data)
        ins_seq, control_flow_relative_offset, masked_ins_idx, masked_ins = zip(*data)
        return BatchedFunctionSeq(arch[0],
                                  ins_seq,
                                  control_flow_relative_offset,
                                  masked_ins_idx,
                                  masked_ins,
                                  predict_classes=20)
