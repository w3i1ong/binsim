import dgl
import torch
import numpy as np
from dgl import DGLGraph
from typing import List, Tuple, Dict, Any
from binsim.disassembly.backend.binaryninja import InsCFG
from binsim.neural.utils.data.dataset.datasetbase import SampleDatasetBase, RandomSamplePairDatasetBase
from binsim.neural.nn.layer.dagnn.dagrnn_ops import create_adj_list
from itertools import chain


class BatchedTokenOperand:
    def __init__(self, register: np.ndarray):
        self._tokens = torch.from_numpy(register)
        self._operand_num = len(self._tokens)

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
    def __init__(self, imm: np.ndarray):
        self._imm = torch.unsqueeze(torch.from_numpy(imm), dim=1)
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

    def astype(self, dtype):
        self._imm = self._imm.to(dtype)


class BatchedX86MemOperand:
    def __init__(self, tokens, disp):
        self._tokens = torch.from_numpy(tokens).reshape([-1, 3])
        self._disp = torch.from_numpy(disp).reshape([-1, 1])

    def to(self, device):
        self._tokens = self._tokens.to(device)
        self._disp = self._disp.to(device)
        return self

    @property
    def operand_num(self):
        return len(self._disp)

    @property
    def tokens(self)->torch.Tensor:
        return self._tokens

    @property
    def disp(self) -> torch.Tensor:
        return self._disp

    def astype(self, dtype):
        self._disp = self._disp.to(dtype)


class BatchedARMMemOperand:
    def __init__(self, tokens: np.ndarray, disp: np.ndarray):
        self._tokens = torch.from_numpy(tokens).reshape([-1, 4])
        self._disp = torch.from_numpy(disp).reshape([-1, 1])

    def to(self, device):
        self._tokens = self._tokens.to(device)
        self._disp = self._disp.to(device)
        return self

    @property
    def operand_num(self):
        return len(self._disp)

    @property
    def tokens(self)->torch.Tensor:
        return self._tokens

    @property
    def disp(self) -> torch.Tensor:
        return self._disp

    def astype(self, dtype):
        self._disp = self._disp.to(dtype)

class BatchedMIPSMemOperand:
    def __init__(self, tokens: np.ndarray, disp: np.ndarray):
        self._tokens = torch.from_numpy(tokens).reshape([-1, 2])
        self._disp = torch.from_numpy(disp).reshape([-1, 1])

    def to(self, device):
        self._tokens = self._tokens.to(device)
        self._disp = self._disp.to(device)
        return self

    @property
    def operand_num(self):
        return len(self._disp)

    @property
    def disp(self) -> torch.Tensor:
        return self._disp

    @property
    def tokens(self) -> torch.Tensor:
        return self._tokens

    def astype(self, dtype):
        self._disp = self._disp.to(dtype)


class BatchedARMRegisterShiftOperand:
    def __init__(self, tokens: np.ndarray):
        self._tokens = torch.from_numpy(tokens).reshape([-1, 3])

    def to(self, device):
        self._tokens = self._tokens.to(device)
        return self

    @property
    def operand_num(self):
        return len(self._tokens)

    @property
    def tokens(self) -> torch.Tensor:
        return self._tokens

class BatchedARMImmShiftOperand:
    def __init__(self, tokens, imm):
        self._tokens = torch.from_numpy(tokens).reshape([-1, 2])
        self._imm = torch.from_numpy(imm).reshape([-1, 1])

    def to(self, device):
        self._tokens = self._tokens.to(device)
        self._imm = self._imm.to(device)
        return self

    @property
    def operand_num(self):
        return len(self._imm)

    @property
    def tokens(self):
        return self._tokens

    @property
    def imm(self) -> torch.Tensor:
        return self._imm

    def astype(self, dtype):
        self._imm = self._imm.to(dtype)


class BatchedARMVectorRegisterIndex:
    def __init__(self, tokens: np.ndarray):
        self._tokens = torch.from_numpy(tokens).reshape([-1, 3])

    def to(self, device):
        self._tokens = self._tokens.to(device)
        return self

    @property
    def operand_num(self):
        return len(self._tokens)

    @property
    def tokens(self) -> torch.Tensor:
        return self._tokens


class BatchedRegListOperand:
    def __init__(self, reg_list, index):
        self._registers = torch.from_numpy(reg_list)
        self._indexes = torch.from_numpy(index)

    def to(self, device):
        self._registers = self._registers.to(device)
        self._indexes = self._indexes.to(device)
        return self

    @property
    def operand_num(self):
        return self._indexes.unique().shape[0]

    @property
    def registers(self) -> torch.Tensor:
        return self._registers

    @property
    def indexes(self) -> torch.Tensor:
        return self._indexes


class BatchedInstruction:
    def __init__(self, data: dict[str, np.ndarray]):
        self._mnemic = None
        self._register = None
        self._special_token = None
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
        self.generate_middle_representation(data)

    def to(self, device):
        self._mnemic = self._mnemic.to(device)
        if self.has_register_operand:
            self._register = self._register.to(device)
        if self.has_token_operand:
            self._special_token = self._special_token.to(device)
        if self.has_imm_operand:
            self._imm = self._imm.to(device)
        if self.has_x86_mem_operand:
            self._x86_mem = self._x86_mem.to(device)
        if self.has_arm_mem_operand:
            self._arm_mem = self._arm_mem.to(device)
        if self.has_mips_mem_operand:
            self._mips_mem = self._mips_mem.to(device)
        if self.has_reg_shift_operand:
            self._reg_shift = self._reg_shift.to(device)
        if self.has_imm_shift_operand:
            self._imm_shift = self._imm_shift.to(device)
        if self.has_reg_index_operand:
            self._reg_index = self._reg_index.to(device)
        if self.has_reg_list_operand:
            self._reg_list = self._reg_list.to(device)
        for key in self._instruction_index:
            self._instruction_index[key] = self._instruction_index[key].to(device)
        for key in self._operand_index:
            self._operand_index[key] = self._operand_index[key].to(device)
        return self

    def astype(self, dtype):
        self._imm.astype(dtype)
        if self.has_arm_mem_operand:
            self._arm_mem.astype(dtype)
        if self.has_x86_mem_operand:
            self._x86_mem.astype(dtype)
        if self.has_mips_mem_operand:
            self._mips_mem.astype(dtype)
        if self.has_imm_shift_operand:
            self._imm_shift.astype(dtype)
        return self

    @property
    def instructions_index(self):
        return self._instruction_index

    @property
    def operand_index(self):
        return self._operand_index

    @property
    def register_operands(self):
        return self._register

    @property
    def has_register_operand(self):
        return self.register_operands is not None


    @property
    def token_operands(self) -> BatchedTokenOperand:
        return self._special_token

    @property
    def has_token_operand(self):
        return self.token_operands is not None

    @property
    def imm_operands(self) -> BatchedImmOperand:
        return self._imm

    @property
    def has_imm_operand(self):
        return self.imm_operands is not None

    @property
    def x86_mem_operands(self) -> BatchedX86MemOperand:
        return self._x86_mem

    @property
    def has_x86_mem_operand(self):
        return self.x86_mem_operands is not None

    @property
    def arm_mem_operands(self) -> BatchedARMMemOperand:
        return self._arm_mem

    @property
    def has_arm_mem_operand(self):
        return self.arm_mem_operands is not None

    @property
    def mips_mem_operands(self) -> BatchedMIPSMemOperand:
        return self._mips_mem

    @property
    def has_mips_mem_operand(self):
        return self.mips_mem_operands is not None

    @property
    def has_mem_operand(self):
        return self.has_x86_mem_operand or self.has_arm_mem_operand or self.has_mips_mem_operand

    @property
    def has_reg_shift_operand(self):
        return self.arm_reg_shift_operands is not None

    @property
    def has_imm_shift_operand(self):
        return self.arm_imm_shift_operands is not None

    @property
    def has_reg_list_operand(self):
        return self.arm_reg_list_operands is not None

    @property
    def has_reg_index_operand(self):
        return self.arm_reg_index_operands is not None

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

    def generate_middle_representation(self, instructions: dict[str, np.ndarray]):
        # mnemonics
        if "mnemonic" in instructions:
            self._mnemic = torch.from_numpy(instructions["mnemonic"])

        # register
        if "register" in instructions:
            self._register = BatchedTokenOperand(instructions["register"])
        # immediate
        if "immediate" in instructions:
            self._imm = BatchedImmOperand(instructions["immediate"])
        # special token
        if "token" in instructions:
            self._special_token = BatchedTokenOperand(instructions["token"])
        # x86 memory
        if "x86_memory.tokens" in instructions:
            self._x86_mem = BatchedX86MemOperand(instructions["x86_memory.tokens"], instructions["x86_memory.disp"])
        # arm memory
        if "arm_memory.tokens" in instructions:
            self._arm_mem = BatchedARMMemOperand(instructions["arm_memory.tokens"], instructions["arm_memory.disp"])
        # mips memory
        if "mips_memory.tokens" in instructions:
            self._mips_mem = BatchedMIPSMemOperand(instructions["mips_memory.tokens"], instructions["mips_memory.disp"])
        # arm register shift
        if "arm_reg_shift" in instructions:
            self._reg_shift = BatchedARMRegisterShiftOperand(instructions["arm_reg_shift"])
        # arm imm shift
        if "arm_imm_shift.tokens" in instructions:
            self._imm_shift = BatchedARMImmShiftOperand(instructions["arm_imm_shift.tokens"], instructions["arm_imm_shift.imm"])
        # arm vector register index
        if "arm_vec_reg" in instructions:
            self._reg_index = BatchedARMVectorRegisterIndex(instructions["arm_vec_reg"])
        # arm register list
        if "reg_list" in instructions:
            self._reg_list = BatchedRegListOperand(instructions["reg_list"], instructions["reg_list.index"])

        self._instruction_index = dict()
        self._operand_index = dict()
        for pos in range(0,10):
            if f"ins_operand.op_idx_{pos}" not in instructions:
                break
            self._instruction_index[pos] = torch.from_numpy(instructions[f"ins_operand.ins_idx_{pos}"])
            self._operand_index[pos] = torch.from_numpy(instructions[f"ins_operand.op_idx_{pos}"])


class BatchedBBIndex:
    def __init__(self, chunks, indexes=None):
        if indexes is None:
            assert len(chunks) == 1
            bb_ins_index, lengths = chunks[0]
            lengths = torch.from_numpy(lengths)
            bb_ins_index = torch.tensor(bb_ins_index)
            sorted_lengths, original_index = torch.sort(lengths, descending=True)
            bb_ins_index = bb_ins_index[original_index]
            _, index = torch.sort(original_index)
            self._old_index = index
            self._bb_index_chunks = [(bb_ins_index, sorted_lengths)]
            self._valid_element_num = sum(sorted_lengths)
            self._padding_element_num = bb_ins_index.numel() - self._valid_element_num
        else:
            _, index = torch.sort(indexes)
            self._old_index = index
            self._bb_index_chunks = [(chunks[i], chunks[i+1]) for i in range(0, len(chunks), 2)]
            self._valid_element_num = sum([sum(lengths) for _, lengths in self._bb_index_chunks])
            self._padding_element_num = sum([bb.numel() for bb, _ in self._bb_index_chunks]) - self._valid_element_num

    @property
    def original_index(self):
        return self._old_index

    @property
    def bb_chunks(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self._bb_index_chunks

    @bb_chunks.setter
    def bb_chunks(self, chunks):
        self._bb_index_chunks = chunks

    def to(self, device):
        for i in range(len(self._bb_index_chunks)):
            chunk, chunk_length = self._bb_index_chunks[i]
            self._bb_index_chunks[i] = (chunk.to(device), chunk_length.to(device))
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
    def collate_fn_py(data: List, **kwargs) -> Tuple[DGLGraph, BatchedBBIndex, BatchedInstruction]:
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

    @staticmethod
    def collate_fn_raw(data, chunks=4, use_dag=True, fast_dag=False, **kwargs):
        from binsim.neural.nn.layer.dagnn.utils.utils import prepare_update_information_for_faster_forward
        if use_dag:
            if chunks <= 1:
                data, edges, (*chunks, node_nums, node_ids)  = (
                    InsCFG.collate_raw_neural_input(data, chunks=chunks, **kwargs))
                indexes = None
            else:
                data, edges, (*chunks, indexes, node_nums, node_ids) = (
                    InsCFG.collate_raw_neural_input(data, chunks=chunks, **kwargs))
            chunks = [torch.from_numpy(chunk) for chunk in chunks]
            indexes = torch.from_numpy(indexes)
            batched_bb_ins_index = BatchedBBIndex(chunks, indexes)
            unique_instructions = BatchedInstruction(data)
            if not fast_dag:
                graphs = []
                for (src, dst), node_nums in zip(edges, node_nums):
                    graphs.append(dgl.graph((src.astype(np.int32), dst.astype(np.int32)), num_nodes=node_nums))
                batched_graph = dgl.batch(graphs)
                batched_graph.ndata["node_id"] = torch.from_numpy(node_ids)
                return batched_graph, batched_bb_ins_index, unique_instructions
            else:
                total_nodes = sum(node_nums)
                adj_list = [[] for _ in range(total_nodes)]
                graph_of_node = []
                base = 0
                for graph_id, ((src_list, dst_list), node_num) in enumerate(zip(edges, node_nums)):
                    for src, dst in zip(src_list, dst_list):
                        adj_list[src + base].append(dst + base)
                    base += node_num
                    graph_of_node.extend([graph_id] * node_num)
                prop_info = prepare_update_information_for_faster_forward(adj_list)
                # prop_info.check(adj_list)
                prop_info.node_ids = torch.from_numpy(node_ids)
                prop_info.graph_ids = torch.tensor(graph_of_node)
                return prop_info, batched_bb_ins_index, unique_instructions

        else:
            res = InsCFG.collate_raw_neural_input(data, chunks=chunks, use_dag=use_dag, **kwargs)
            data, chunks, (indexes, node2seq, masks) = res
            unique_instructions = BatchedInstruction(data)
            chunks = [torch.from_numpy(chunk).to(torch.int32) for chunk in chain.from_iterable(chunks)]
            indexes = torch.from_numpy(indexes)
            batched_bb_ins_index = BatchedBBIndex(chunks, indexes)
            node2seq = torch.from_numpy(node2seq)
            masks = torch.from_numpy(masks)
            return unique_instructions, batched_bb_ins_index, node2seq, masks



class InsCFGSamplePairDataset(RandomSamplePairDatasetBase):
    def __init__(self, data: Dict[Any, List[InsCFG]], sample_format=None, expand_time=None, **kwargs):
        super().__init__(data, sample_format=sample_format, expand_time=expand_time, **kwargs)

    @property
    def SingleSampleClass(self):
        return InsCFGSampleDataset
