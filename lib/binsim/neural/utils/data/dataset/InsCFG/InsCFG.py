from binsim.disassembly.binaryninja import InsCFG
from typing import List, Tuple, Dict, Any, Union, Set
from dgl import DGLGraph
import dgl
import torch
from binsim.neural.utils.data.dataset.datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from .utils import basic_block_length_chunk
from itertools import chain
from binsim.disassembly.binaryninja.core.graph import BSInstruction, X86MemOperand, ARMMemOperand, MIPSMemOperand, \
    REGOperand, IMMOperand, SpecialTokenOperand, ARMRegisterShiftOperand, ARMImmShiftOperand, REGListOperand, \
    ARMVectorRegisterIndex


class BatchedTokenOperand:
    def __init__(self, tokens: Set[Union[REGOperand, SpecialTokenOperand]]):
        index_list = [token.reg if isinstance(token, REGOperand) else token.token
                      for token in tokens]
        self._tokens = torch.tensor(index_list, dtype=torch.long)
        self._operand_num = len(tokens)

    def to(self, device):
        self._tokens = self._tokens.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def tokens(self) -> torch.Tensor:
        return self._tokens


class BatchedImmOperand:
    def __init__(self, imm: Set[IMMOperand]):
        imm = [val.imm for val in imm]
        self._imm = torch.unsqueeze(torch.tensor(imm, dtype=torch.float32), dim=1)
        self._operand_num = len(imm)

    def to(self, device):
        self._imm = self._imm.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def imm(self) -> torch.Tensor:
        return self._imm


class BatchedX86MemOperand:
    def __init__(self, mem: Set[X86MemOperand]):
        base_registers = [operand.base for operand in mem]
        index_registers = [operand.index for operand in mem]
        scale_values = [operand.scale for operand in mem]
        disp_values = [operand.disp for operand in mem]

        self._base = torch.tensor(base_registers, dtype=torch.long)
        self._index = torch.tensor(index_registers, dtype=torch.long)
        self._scale = torch.tensor(scale_values, dtype=torch.float32)
        self._disp = torch.tensor(disp_values, dtype=torch.float32)
        self._operand_num = len(mem)

    def to(self, device):
        self._base = self._base.to(device)
        self._index = self._index.to(device)
        self._scale = self._scale.to(device)
        self._disp = self._disp.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def base(self) -> torch.Tensor:
        return self._base

    @property
    def index(self) -> torch.Tensor:
        return self._index

    @property
    def scale(self) -> torch.Tensor:
        return self._scale

    @property
    def disp(self) -> torch.Tensor:
        return self._disp


class BatchedARMMemOperand:
    def __init__(self, mem: Set[ARMMemOperand]):
        base_register = [operand.base for operand in mem]
        index_register = [operand.index for operand in mem]
        shift_type = [operand.shift_type for operand in mem]
        shift_value = [operand.shift_value for operand in mem]
        disp_values = [operand.disp for operand in mem]

        self._base = torch.tensor(base_register)
        self._index = torch.tensor(index_register)
        self._shift_type = torch.tensor(shift_type)
        self._shift_value = torch.tensor(shift_value)
        self._disp = torch.tensor(disp_values, dtype=torch.float32)
        self._operand_num = len(mem)

    def to(self, device):
        self._base = self._base.to(device)
        self._index = self._index.to(device)
        self._shift_type = self._shift_type.to(device)
        self._shift_value = self._shift_value.to(device)
        self._disp = self._disp.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def base(self) -> torch.Tensor:
        return self._base

    @property
    def index(self) -> torch.Tensor:
        return self._index

    @property
    def shift_type(self) -> torch.Tensor:
        return self._shift_type

    @property
    def shift_value(self) -> torch.Tensor:
        return self._shift_value

    @property
    def disp(self) -> torch.Tensor:
        return self._disp


class BatchedMIPSMemOperand:
    def __init__(self, mem: Set[MIPSMemOperand]):
        base_register = [operand.base for operand in mem]
        index_register = [operand.index for operand in mem]
        disp_values = [operand.disp for operand in mem]

        self._base = torch.tensor(base_register)
        self._index = torch.tensor(index_register)
        self._disp = torch.tensor(disp_values, dtype=torch.float32)
        self._operand_num = len(mem)

    def to(self, device):
        self._base = self._base.to(device)
        self._index = self._index.to(device)
        self._disp = self._disp.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def base(self) -> torch.Tensor:
        return self._base

    @property
    def disp(self) -> torch.Tensor:
        return self._disp

    @property
    def index(self) -> torch.Tensor:
        return self._index


class BatchedARMRegisterShiftOperand:
    def __init__(self, operands: Set[ARMRegisterShiftOperand]):
        shift_type = [operand.shift_type for operand in operands]
        value = [operand.value for operand in operands]
        register = [operand.register for operand in operands]

        self._shift_type = torch.tensor(shift_type, dtype=torch.long)
        self._value = torch.tensor(value, dtype=torch.long)
        self._register = torch.tensor(register, dtype=torch.long)
        self._operand_num = len(operands)

    def to(self, device):
        self._shift_type = self._shift_type.to(device)
        self._value = self._value.to(device)
        self._register = self._register.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def shift_type(self) -> torch.Tensor:
        return self._shift_type

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @property
    def register(self) -> torch.Tensor:
        return self._register


class BatchedARMImmShiftOperand:
    def __init__(self, operands: Set[ARMImmShiftOperand]):
        shift_type = [operand.shift_type for operand in operands]
        shift_value = [operand.value for operand in operands]
        imm_value = [operand.imm for operand in operands]

        self._shift_type = torch.tensor(shift_type, dtype=torch.long)
        self._value = torch.tensor(shift_value, dtype=torch.long)
        self._imm = torch.tensor(imm_value, dtype=torch.float)
        self._operand_num = len(operands)

    def to(self, device):
        self._shift_type = self._shift_type.to(device)
        self._value = self._value.to(device)
        self._imm = self._imm.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def shift_type(self) -> torch.Tensor:
        return self._shift_type

    @property
    def value(self) -> torch.Tensor:
        return self._value

    @property
    def imm(self) -> torch.Tensor:
        return self._imm


class BatchedARMVectorRegisterIndex:
    def __init__(self, operands: Set[ARMVectorRegisterIndex]):
        register = [operand.register for operand in operands]
        index = [operand.index for operand in operands]
        vector_type = [operand.type for operand in operands]

        self._register = torch.tensor(register, dtype=torch.long)
        self._index = torch.tensor(index, dtype=torch.long)
        self._type = torch.tensor(vector_type, dtype=torch.long)
        self._operand_num = len(operands)

    def to(self, device):
        self._register = self._register.to(device)
        self._index = self._index.to(device)
        self._type = self._type.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def register(self) -> torch.Tensor:
        return self._register

    @property
    def index(self) -> torch.Tensor:
        return self._index

    @property
    def type(self) -> torch.Tensor:
        return self._type


class BatchedRegListOperand:
    def __init__(self, regs: Set[REGListOperand]):
        registers, indexes = [], []
        for index, reg_list in enumerate(regs):
            registers.extend(reg_list.regs)
            indexes.extend([index] * len(reg_list.regs))
        self._registers = torch.tensor(registers)
        self._indexes = torch.tensor(indexes)
        self._operand_num = len(regs)

    def to(self, device):
        self._registers = self._registers.to(device)
        self._indexes = self._indexes.to(device)
        return self

    @property
    def operand_num(self):
        return self._operand_num

    @property
    def registers(self) -> torch.Tensor:
        return self._registers

    @property
    def indexes(self) -> torch.Tensor:
        return self._indexes


class BatchedInstruction:
    def __init__(self, instructions):
        self._mnemic = None
        self._register = None
        self._imm = None
        self._x86_mem = None
        self._arm_mem = None
        self._mips_mem = None
        self._reg_shift = None
        self._imm_shift = None
        self._reg_list = None
        self._reg_index = None
        self._instruction_index = None
        self._operand_index = None
        # self._instructions = instructions
        self.generate_middle_representation(instructions)

    def to(self, device):
        self._mnemic = self._mnemic.to(device)
        self._register = self._register.to(device)
        self._imm = self._imm.to(device)
        self._x86_mem = self._x86_mem.to(device)
        self._arm_mem = self._arm_mem.to(device)
        self._mips_mem = self._mips_mem.to(device)
        self._reg_shift = self._reg_shift.to(device)
        self._reg_list = self._reg_list.to(device)
        self._reg_index = self._reg_index.to(device)
        for key in self._instruction_index:
            self._instruction_index[key] = self._instruction_index[key].to(device)
        for key in self._operand_index:
            self._operand_index[key] = tuple(t.to(device) for t in self._operand_index[key])
        return self

    @property
    def instructions_index(self):
        return self._instruction_index

    @property
    def operand_index(self):
        return self._operand_index

    @property
    def has_token_operand(self):
        return self.token_operands.operand_num > 0

    @property
    def has_imm_operand(self):
        return self.imm_operands.operand_num > 0

    @property
    def has_mem_operand(self):
        return self.x86_mem_operands.operand_num > 0 or \
            self.arm_mem_operands.operand_num > 0 or \
            self.mips_mem_operands.operand_num > 0

    @property
    def has_reg_shift_operand(self):
        return self.arm_reg_shift_operands.operand_num > 0

    @property
    def has_imm_shift_operand(self):
        return self.arm_imm_shift_operands.operand_num > 0

    @property
    def has_reg_list_operand(self):
        return self.arm_reg_list_operands.operand_num > 0

    @property
    def has_reg_index_operand(self):
        return self.arm_reg_index_operands.operand_num > 0

    @property
    def token_operands(self) -> BatchedTokenOperand:
        return self._register

    @property
    def imm_operands(self) -> BatchedImmOperand:
        return self._imm

    @property
    def x86_mem_operands(self) -> BatchedX86MemOperand:
        return self._x86_mem

    @property
    def arm_mem_operands(self) -> BatchedARMMemOperand:
        return self._arm_mem

    @property
    def mips_mem_operands(self) -> BatchedMIPSMemOperand:
        return self._mips_mem

    @property
    def mnemic(self):
        return self._mnemic

    @property
    def arm_reg_shift_operands(self) -> BatchedARMRegisterShiftOperand:
        return self._reg_shift

    @property
    def arm_imm_shift_operands(self) -> BatchedARMImmShiftOperand:
        return self._imm_shift

    @property
    def arm_reg_list_operands(self) -> BatchedRegListOperand:
        return self._reg_list

    @property
    def arm_reg_index_operands(self) -> BatchedARMVectorRegisterIndex:
        return self._reg_index

    @staticmethod
    def _gen_operand2idx(operands, base=0) -> dict:
        return {operand: idx + base for idx, operand in enumerate(operands)}

    def generate_middle_representation(self, instructions: List[BSInstruction]):
        # deal with mnemic
        mnemic = [ins.mnemonic for ins in instructions]
        # deal with operands
        # 1. get unique operands
        imm_operands, token_operands = set(), set()
        reg_shift_operands, reg_list_operands, imm_shift_operands = set(), set(), set()
        x86_mem_operands, arm_mem_operands, mips_mem_operands = set(), set(), set()
        reg_index_operands = set()
        for ins in instructions:
            for operand in ins.operands:
                if isinstance(operand, (REGOperand, SpecialTokenOperand)):
                    token_operands.add(operand)
                elif isinstance(operand, IMMOperand):
                    imm_operands.add(operand)
                elif isinstance(operand, X86MemOperand):
                    x86_mem_operands.add(operand)
                elif isinstance(operand, ARMMemOperand):
                    arm_mem_operands.add(operand)
                elif isinstance(operand, MIPSMemOperand):
                    mips_mem_operands.add(operand)
                elif isinstance(operand, ARMRegisterShiftOperand):
                    reg_shift_operands.add(operand)
                elif isinstance(operand, ARMImmShiftOperand):
                    imm_shift_operands.add(operand)
                elif isinstance(operand, REGListOperand):
                    reg_list_operands.add(operand)
                elif isinstance(operand, ARMVectorRegisterIndex):
                    reg_index_operands.add(operand)
                else:
                    raise NotImplementedError(f"Unsupported operand type {type(operand)}")
        # 2. generate index for each operand.
        base = 0
        token_operand2idx = self._gen_operand2idx(token_operands, base)
        base += len(token_operand2idx)
        imm_operand2idx = self._gen_operand2idx(imm_operands, base)
        base += len(imm_operand2idx)
        x86_mem_operand2idx = self._gen_operand2idx(x86_mem_operands, base)
        base += len(x86_mem_operand2idx)
        arm_mem_operand2idx = self._gen_operand2idx(arm_mem_operands, base)
        base += len(arm_mem_operand2idx)
        mips_mem_operand2idx = self._gen_operand2idx(mips_mem_operands, base)
        base += len(mips_mem_operand2idx)
        reg_shift_operand2idx = self._gen_operand2idx(reg_shift_operands, base)
        base += len(reg_shift_operand2idx)
        imm_shift_operand2idx = self._gen_operand2idx(imm_shift_operands, base)
        base += len(imm_shift_operand2idx)
        reg_list_operand2idx = self._gen_operand2idx(reg_list_operands, base)
        base += len(reg_list_operand2idx)
        reg_index_operand2idx = self._gen_operand2idx(reg_index_operands, base)
        base += len(reg_index_operand2idx)
        operand2idx = {
            REGOperand: token_operand2idx,
            SpecialTokenOperand: token_operand2idx,
            IMMOperand: imm_operand2idx,
            X86MemOperand: x86_mem_operand2idx,
            ARMMemOperand: arm_mem_operand2idx,
            MIPSMemOperand: mips_mem_operand2idx,
            ARMRegisterShiftOperand: reg_shift_operand2idx,
            ARMImmShiftOperand: imm_shift_operand2idx,
            REGListOperand: reg_list_operand2idx,
            ARMVectorRegisterIndex: reg_index_operand2idx
        }

        self._register = BatchedTokenOperand(token_operands)
        self._imm = BatchedImmOperand(imm_operands)
        self._x86_mem = BatchedX86MemOperand(x86_mem_operands)
        self._arm_mem = BatchedARMMemOperand(arm_mem_operands)
        self._mips_mem = BatchedMIPSMemOperand(mips_mem_operands)
        self._reg_shift = BatchedARMRegisterShiftOperand(reg_shift_operands)
        self._imm_shift = BatchedARMImmShiftOperand(imm_shift_operands)
        self._reg_list = BatchedRegListOperand(reg_list_operands)
        self._reg_index = BatchedARMVectorRegisterIndex(reg_index_operands)

        #  generate middle representation for mnemonic
        instruction_index, operand_index = {}, {}
        for i in range(6):
            instruction_index[i] = []
            operand_index[i] = []
        for ins_idx, instr in enumerate(instructions):
            for operand_position, operand in enumerate(instr.operands):
                operand_index[operand_position].append(operand2idx[type(operand)][operand])
                instruction_index[operand_position].append(ins_idx)
        for key in instruction_index.keys():
            instruction_index[key] = torch.tensor(instruction_index[key])
            operand_index[key] = self.compress_operand_index(operand_index[key])

        self._operand_index = operand_index
        self._instruction_index = instruction_index
        self._mnemic = torch.tensor(mnemic, dtype=torch.long)

    def compress_operand_index(self, indexes: List[int]):
        unique_indexes = list(set(indexes))
        index_map = {index: i for i, index in enumerate(unique_indexes)}
        new_indexes = [index_map[index] for index in indexes]
        return torch.tensor(unique_indexes), torch.tensor(new_indexes)


class BatchedBBIndex:
    def __init__(self, bb_ins_index, lengths=None, k=10, max_padding_ratio=2):
        self._old_index = None
        self._bb_index_chunks = None
        self._valid_element_num = 0
        self._padding_element_num = 0
        self.init(bb_ins_index, lengths, k, max_padding_ratio)

    def basic_block_length_chunk(self, lengths, k, max_padding_ratio, check_sorted_decrease=True):
        chunk_end_list = basic_block_length_chunk(lengths[::-1], k, max_padding_ratio, need_check=check_sorted_decrease)
        return [len(lengths) - end_idx for end_idx in chunk_end_list[::-1]] + [len(lengths)]

    def init(self, bb_ins_index, lengths, k, max_padding_ratio):
        if lengths is None:
            lengths = [len(bb) for bb in bb_ins_index]
        lengths = torch.tensor(lengths)
        sorted_lengths, original_index = torch.sort(lengths, descending=True)
        bb_ins_index = [bb_ins_index[idx] for idx in original_index]
        _, index = torch.sort(original_index)
        self._old_index = index
        sorted_lengths = list(sorted_lengths.numpy())
        chunks_end_list = self.basic_block_length_chunk(sorted_lengths, k, max_padding_ratio,
                                                        check_sorted_decrease=False)
        chunk_start = 0
        self._bb_index_chunks = []
        for chunk_end in chunks_end_list:
            bb_chunk, max_length = [], sorted_lengths[chunk_start]
            for ins_idx in range(chunk_start, chunk_end):
                bb_chunk.append(bb_ins_index[ins_idx] + [0] * (max_length - sorted_lengths[ins_idx]))
            bb_chunk = torch.tensor(bb_chunk)
            bb_chunk_length = torch.tensor(sorted_lengths[chunk_start:chunk_end])
            self._bb_index_chunks.append((bb_chunk, bb_chunk_length))
            chunk_start = chunk_end
        self._valid_element_num = sum(sorted_lengths)
        self._padding_element_num = sum(
            [bb.shape[0] * bb.shape[1] for bb, bb_len in self._bb_index_chunks]) - self._valid_element_num

    @property
    def original_index(self):
        return self._old_index

    @property
    def bb_chunks(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self._bb_index_chunks

    def to(self, device):
        for i in range(len(self._bb_index_chunks)):
            chunk, chunk_length = self._bb_index_chunks[i]
            self._bb_index_chunks[i] = (chunk.to(device), chunk_length)
        return self

    @property
    def valid_element_num(self):
        return self._valid_element_num

    @property
    def padding_element_num(self):
        return self._padding_element_num


class InsCFGSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[InsCFG],
                 samples_id: torch.Tensor = None,
                 tags: List[Any] = None,
                 with_name=False, **kwargs):
        super().__init__(data, sample_id=samples_id, with_name=with_name, tags=tags, **kwargs)

    @staticmethod
    def collate_fn(data: List) -> Tuple[DGLGraph, BatchedBBIndex, BatchedInstruction]:
        graphs, basic_blocks, lengths = zip(*data)
        # build graph
        batched_graph = SampleDatasetBase.batch_graph(graphs)

        basic_blocks = [bb for function in basic_blocks for bb in function]
        lengths = [length for bb_length_list in lengths for length in bb_length_list]

        unique_instructions = list(set(chain.from_iterable(basic_blocks)))
        instruction2idx = {instr: i for i, instr in enumerate(unique_instructions)}

        # build the instruction index matrix for each basic block
        bb_ins_index = [None] * len(basic_blocks)
        for bb_index, (basic_block, length) in enumerate(zip(basic_blocks, lengths)):
            basic_block_ins_indexes = [instruction2idx[ins] for ins in basic_block]
            bb_ins_index[bb_index] = basic_block_ins_indexes
        batched_bb_ins_index = BatchedBBIndex(bb_ins_index)
        unique_instructions = BatchedInstruction(unique_instructions)
        return batched_graph, batched_bb_ins_index, unique_instructions


class InsCFGSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[InsCFG]], sample_format=None, expand_time=None, **kwargs):
        super().__init__(data, sample_format=sample_format, expand_time=expand_time, **kwargs)

    @property
    def SingleSampleClass(self):
        return InsCFGSampleDataset


class InsSeqSampleDataset(SampleDatasetBase):
    def __init__(self, data: List[InsCFG],
                 samples_id: torch.Tensor,
                 tags: List[Any],
                 with_name=False, **kwargs):
        super().__init__(data, sample_id=samples_id, with_name=with_name, tags=tags, **kwargs)

    @staticmethod
    def collate_fn(functions: List[List]) -> Tuple[BatchedInstruction, torch.Tensor, torch.Tensor]:
        lengths = [len(function) for function in functions]

        unique_instructions = list(set(chain.from_iterable(functions)))
        instruction2idx = {instr: i for i, instr in enumerate(unique_instructions)}

        # build the instruction index matrix for each basic block
        func_ins_index = [None] * len(lengths)
        max_length = max(map(len, functions))
        for func_index, (function, length) in enumerate(zip(functions, lengths)):
            basic_block_ins_indexes = [instruction2idx[ins] for ins in function] + [0] * (max_length - len(function))
            func_ins_index[func_index] = basic_block_ins_indexes
        func_ins_index = torch.Tensor(func_ins_index).long()
        unique_instructions = BatchedInstruction(unique_instructions)
        return unique_instructions, func_ins_index, torch.tensor(lengths)


class InsSeqSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[InsCFG]], sample_format=None, expand_time=None, **kwargs):
        super().__init__(data, sample_format=sample_format, expand_time=expand_time, **kwargs)

    @property
    def SingleSampleClass(self):
        return InsSeqSampleDataset
