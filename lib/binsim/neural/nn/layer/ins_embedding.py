import torch
import typing
from torch import nn
from typing import List

if typing.TYPE_CHECKING:
    from binsim.neural.utils.data import BatchedInstruction


class ImmEmbedding(nn.Module):
    def __init__(self, embed_dim=128, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self._embed_dim = embed_dim
        self._layers = nn.Sequential(
            nn.Linear(1, embed_dim, **factory_kwargs),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, embed_dim, **factory_kwargs),
            nn.LeakyReLU(0.1)
        )

    def forward(self, imm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sign = torch.sign(imm)
            mod = torch.abs(imm)
            value = torch.log2(mod) * sign
            imm = torch.where(torch.gt(mod, 2), value, imm)
        return self._layers(imm)


class MultiTokenOperandEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim, layer_num=1, dtype=None, device=None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self._in_dim = in_dim
        self._embed_dim = embed_dim
        layers = [nn.Linear(in_dim, embed_dim, True, **factory_kwargs)]
        for _ in range(layer_num):
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs))
        layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor):
        return self.layers(tokens)


class InstructionEmbedding(nn.Module):
    def __init__(self, vocab_num, embed_dim=128, place_holders=0, device=None, dtype=None):
        """
        A simple neural network model used to generate instruction embedding for disassembly instruction.
        :param vocab_num: The size of vocabulary.
        :param embed_dim: The dimension of the instruction embedding.
        :param device: The device used to store the weights.
        :param dtype: The data type used to store the weights.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._embed_dim = embed_dim
        self._token_embedding = nn.Embedding(vocab_num, embed_dim, **factory_kwargs)

        self._imm_embedding = ImmEmbedding(embed_dim=embed_dim, **factory_kwargs)
        self._x86_mem_embedding_aggregator = MultiTokenOperandEmbedding(4 * embed_dim, embed_dim,
                                                                        **factory_kwargs)
        self._arm_mem_embedding_aggregator = MultiTokenOperandEmbedding(5 * embed_dim, embed_dim,
                                                                        **factory_kwargs)
        self._mips_mem_embedding_aggregator = MultiTokenOperandEmbedding(3 * embed_dim, embed_dim,
                                                                         **factory_kwargs)
        self._arm_shift_embedding_aggregator = MultiTokenOperandEmbedding(3 * embed_dim, embed_dim,
                                                                          **factory_kwargs)
        self._reg_index_embedding_aggregator = MultiTokenOperandEmbedding(3 * embed_dim, embed_dim,
                                                                          **factory_kwargs)
        self._operand_linear = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, **factory_kwargs) for _ in range(6)]
        )
        self._instruction_embedding_aggregator = nn.Sequential(
            nn.LeakyReLU(0.1), nn.Linear(embed_dim, embed_dim, **factory_kwargs), nn.LeakyReLU(0.1)
        )

        if place_holders == 0:
            self._place_holders = None
        else:
            self._place_holders = torch.nn.Parameter(torch.randn(place_holders, embed_dim, **factory_kwargs, requires_grad=True))

    def _arm_reg_shift_embedding(self, instruction: 'BatchedInstruction') -> torch.Tensor:
        operands = instruction.arm_reg_shift_operands
        tokens = self._token_embedding(operands.tokens)
        return self._arm_shift_embedding_aggregator(tokens.reshape([-1, 3 * self._embed_dim]))

    def _arm_imm_shift_embedding(self, instruction: 'BatchedInstruction') -> torch.Tensor:
        operands = instruction.arm_imm_shift_operands
        imm = self._imm_embedding(operands.imm.reshape([-1, 1, 1]))
        tokens = self._token_embedding(operands.tokens)
        tokens = torch.cat([tokens, imm], dim=1)
        return self._arm_shift_embedding_aggregator(tokens.reshape([-1, 3 * self._embed_dim]))

    def _arm_reg_list_embedding(self, instruction: 'BatchedInstruction') -> torch.Tensor:
        operands = instruction.arm_reg_list_operands
        reg_list, indexes, operand_num = operands.registers, operands.indexes, operands.operand_num
        reg_list = self._token_embedding(reg_list)
        embed_dim = reg_list.shape[-1]
        vec_sum = torch.zeros([operand_num, embed_dim], device=reg_list.device, dtype=reg_list.dtype)
        vec_sum.index_add_(0, indexes, reg_list)
        return vec_sum

    def _arm_reg_index_embedding(self, instruction: 'BatchedInstruction') -> torch.Tensor:
        operands = instruction.arm_reg_index_operands
        tokens = self._token_embedding(operands.tokens).reshape([-1, 3 * self._embed_dim])
        return self._reg_index_embedding_aggregator(tokens)

    def forward(self, instruction: 'BatchedInstruction'):
        return self.normal_forward(instruction)

    def normal_forward(self, instruction: 'BatchedInstruction'):
        # 1. generate embedding for Memory Operand, token operand and immediate operand
        operands = []

        if instruction.has_register_operand:
            register_operands = self._token_embedding(instruction.register_operands.tokens)
            operands.append(register_operands)

        if instruction.has_token_operand:
            token_operands = self._token_embedding(instruction.token_operands.tokens)
            operands.append(token_operands)

        if instruction.has_imm_operand:
            imm_operands = self._imm_embedding(instruction.imm_operands.imm)
            operands.append(imm_operands)

        if instruction.has_mem_operand:
            mem_operands = self._mem_embedding(instruction)
            operands.extend(mem_operands)

        if instruction.has_reg_shift_operand:
            reg_shift_operands = self._arm_reg_shift_embedding(instruction)
            operands.append(reg_shift_operands)

        if instruction.has_imm_shift_operand:
            imm_shift_operands = self._arm_imm_shift_embedding(instruction)
            operands.append(imm_shift_operands)

        if instruction.has_reg_index_operand:
            reg_index = self._arm_reg_index_embedding(instruction)
            operands.append(reg_index)

        if instruction.has_reg_list_operand:
            reg_list = self._arm_reg_list_embedding(instruction)
            operands.append(reg_list)

        operands = torch.cat(operands, dim=0)
        # 2. generate embedding for mnemonic
        mnemonic_embeddings = self._token_embedding(instruction.mnemic)
        # 3. generate embedding for instruction
        instructions_index = instruction.instructions_index
        operands_index = instruction.operand_index
        for key in instructions_index:
            operand_to_ins_index = instructions_index[key]
            # calculate weight matrix for operands
            operands_embeddings = operands[operands_index[key]]
            operands_embeddings = self._operand_linear[key](operands_embeddings)
            # calculate instruction embedding
            mnemonic_embeddings.index_add_(0, operand_to_ins_index, operands_embeddings)
        ins_embeddings = self._instruction_embedding_aggregator(mnemonic_embeddings)
        if self._place_holders is not None:
            ins_embeddings = torch.cat([ins_embeddings, self._place_holders], dim=0)
        return ins_embeddings

    def _mem_embedding(self, instruction: 'BatchedInstruction') -> List[torch.Tensor]:
        result = []
        # x86
        if instruction.has_x86_mem_operand:
            x86_mem_operands = instruction.x86_mem_operands
            token_index = x86_mem_operands.tokens
            disp = x86_mem_operands.disp.reshape([-1, 1, 1])
            token_embedding = self._token_embedding(token_index)
            disp_embedding = self._imm_embedding(disp)
            x86_mem_embedding = torch.cat([token_embedding, disp_embedding], dim=1)
            result.append(self._x86_mem_embedding_aggregator(x86_mem_embedding.reshape([-1, 4 * self._embed_dim])))

        # arm
        if instruction.has_arm_mem_operand:
            arm_operands = instruction.arm_mem_operands
            token_embedding = self._token_embedding(arm_operands.tokens)
            disp_embedding = self._imm_embedding(arm_operands.disp.reshape([-1, 1, 1]))
            arm_mem_embedding = torch.cat([token_embedding, disp_embedding], dim=1)
            result.append(self._arm_mem_embedding_aggregator(arm_mem_embedding.reshape([-1, 5 * self._embed_dim])))

        # mips
        if instruction.has_mips_mem_operand:
            mips_operands = instruction.mips_mem_operands
            token_embedding = self._token_embedding(mips_operands.tokens)
            imm_embedding = self._imm_embedding(mips_operands.disp.reshape([-1, 1, 1]))
            mips_mem_embedding = torch.cat([token_embedding, imm_embedding], dim=1)
            result.append(self._mips_mem_embedding_aggregator(mips_mem_embedding.reshape([-1, 3 * self._embed_dim])))

        return result

    @property
    def token_embedding(self):
        return self._token_embedding
