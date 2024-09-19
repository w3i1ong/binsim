from collections import namedtuple
from enum import Enum
from dataclasses import dataclass
from typing import Any, Union, List


class BSInsOperandType(Enum):
    IMM = 1
    REG = 2
    ARM_MEM = 3
    X86_MEM = 4
    MIPS_MEM = 5
    REG_LIST = 6
    SPECIAL_TOKEN = 7


X86MemOperand = namedtuple('X86MemOperand', ['base', 'index', 'scale', 'disp', 'bits'])
ARMMemOperand = namedtuple('ARMMemOperand', ['base', 'index', 'shift_type', 'shift_value', 'disp'])
MIPSMemOperand = namedtuple('MIPSMemOperand', ['base', 'disp', 'index', 'bits'])
REGListOperand = namedtuple('REGListOperand', ['regs'])
REGOperand = namedtuple('REGOperand', ['reg'])
IMMOperand = namedtuple('IMMOperand', ['imm'])
SpecialTokenOperand = namedtuple('SpecialTokenOperand', ['token'])
ARMRegisterShiftOperand = namedtuple('ARMRegisterShiftOperand', ['shift_type', 'value', 'register'])
ARMImmShiftOperand = namedtuple('ARMImmShiftOperand', ['shift_type', 'value', 'imm'])
ARMVectorRegisterIndex = namedtuple('ARMVectorRegisterIndex', ['register', 'index', 'type'])

@dataclass
class BSInsOperand:
    type: BSInsOperandType
    value: Any


@dataclass
class BSInstruction:
    mnemonic: Union[str, int]
    operands: List[Any]

    def __eq__(self, other):
        return self.mnemonic == other.mnemonic and self.operands == other.operands

    def __hash__(self):
        return hash((self.mnemonic, tuple(self.operands)))
