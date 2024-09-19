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

class OperandType(Enum):
    X86MemOperand = 1
    ARMMemOperand = 2
    MIPSMemOperand = 3
    REGListOperand = 4
    REGOperand = 5
    IMMOperand = 6
    SpecialTokenOperand = 7
    ARMRegisterShiftOperand = 8
    ARMImmShiftOperand = 9
    ARMVectorRegisterIndex = 10

def X86MemOperand(*, base, index, scale, disp, bits):
    return [OperandType.X86MemOperand, base, index, scale, disp]

def ARMMemOperand(*, base, index, shift_type, shift_value, disp):
    return [OperandType.ARMMemOperand, base, index, shift_type, shift_value, disp]

def MIPSMemOperand(*, base, disp, index, bits):
    return [OperandType.MIPSMemOperand, base, index, disp]

def REGListOperand(*, regs):
    return [OperandType.REGListOperand, len(regs), *regs]

def REGOperand(*, reg):
    return [OperandType.REGOperand, reg]

def IMMOperand(*, imm):
    return [OperandType.IMMOperand, imm]

def SpecialTokenOperand(*, token):
    return [OperandType.SpecialTokenOperand, token]

def ARMRegisterShiftOperand(*, register, shift_type, value):
    return [OperandType.ARMRegisterShiftOperand, register, shift_type, value]

def ARMImmShiftOperand(*, imm, shift_type, value):
    return [OperandType.ARMImmShiftOperand, imm, shift_type, value]

def ARMVectorRegisterIndex(*, register, index, vec_type):
    return [OperandType.ARMVectorRegisterIndex, register, index, vec_type]

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
