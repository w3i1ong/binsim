import torch
from torch import nn
from binsim.neural.utils.data import BatchedInstruction
from typing import List


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
    def __init__(self, vocab_num, embed_dim=128, device=None, dtype=None, mode='normal'):
        """
        A simple neural network model used to generate instruction embedding for disassembly instruction.
        :param vocab_num: The size of vocabulary.
        :param embed_dim: The dimension of the instruction embedding.
        :param device: The device used to store the weights.
        :param dtype: The data type used to store the weights.
        :param mode: The mode used to generate instruction embedding. Can be 'normal', 'mnemonic' or 'mean'.
            If set to 'normal', the embedding model will consider the hierarchy structure of the instruction AST.
            If set to 'mnemonic', mnemonic embedding will be used as the instruction embedding.
            If set to 'mean', the instruction embedding will consist of two parts:
                1. the mnemonic embedding
                2. the mean of the operand token embeddings
                Just like the method used in Asm2Vec.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self._embed_dim = embed_dim
        self._token_embedding = nn.Embedding(vocab_num, embed_dim, **factory_kwargs)
        self._mode = mode
        match mode:
            case 'normal':
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
            case 'mnemonic':
                # token embedding is enough.
                pass
            case 'mean':
                self._imm_embedding = ImmEmbedding(embed_dim=embed_dim, **factory_kwargs)

    def _arm_reg_shift_embedding(self, instruction: BatchedInstruction) -> torch.Tensor:
        operands = instruction.arm_reg_shift_operands
        reg, shift_type, shift_value = operands.register, operands.shift_type, operands.value
        tokens = torch.stack([reg, shift_type, shift_value], dim=1)
        tokens = self._token_embedding(tokens)
        if self._mode == 'normal':
            return self._arm_shift_embedding_aggregator(tokens.reshape([-1, 3 * self._embed_dim]))
        elif self._mode == 'mean':
            return torch.mean(tokens, dim=1)
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

    def _arm_imm_shift_embedding(self, instruction: BatchedInstruction) -> torch.Tensor:
        operands = instruction.arm_imm_shift_operands
        imm, shift_type, shift_value = operands.imm, operands.shift_type, operands.value
        imm = self._imm_embedding(imm.reshape([-1, 1, 1]))
        tokens = torch.stack([shift_type, shift_value], dim=1)
        tokens = self._token_embedding(tokens)
        tokens = torch.cat([tokens, imm], dim=1)
        if self._mode == 'normal':
            return self._arm_shift_embedding_aggregator(tokens.reshape([-1, 3 * self._embed_dim]))
        elif self._mode == 'mean':
            return torch.mean(tokens, dim=1)
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

    def _arm_reg_list_embedding(self, instruction: BatchedInstruction) -> torch.Tensor:
        operands = instruction.arm_reg_list_operands
        reg_list, indexes, operand_num = operands.registers, operands.indexes, operands.operand_num
        reg_list = self._token_embedding(reg_list)
        embed_dim = reg_list.shape[-1]
        vec_sum = torch.zeros([operand_num, embed_dim], device=reg_list.device, dtype=reg_list.dtype)
        vec_sum.index_add_(0, indexes, reg_list)
        if self._mode == 'normal':
            return vec_sum
        elif self._mode == 'mean':
            cnt = torch.zeros([operand_num, 1], device=reg_list.device, dtype=reg_list.dtype)
            ones = torch.ones([reg_list.shape[0], 1], device=reg_list.device, dtype=reg_list.dtype)
            cnt.index_add_(0, indexes, ones)
            return vec_sum / cnt
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

    def _arm_reg_index_embedding(self, instruction: BatchedInstruction) -> torch.Tensor:
        operands = instruction.arm_reg_index_operands
        reg, index, type = operands.register, operands.index, operands.type
        tokens = torch.stack([reg, index, type], dim=1)
        if self._mode == 'normal':
            tokens = self._token_embedding(tokens).reshape([-1, 3 * self._embed_dim])
            return self._reg_index_embedding_aggregator(tokens)
        elif self._mode == 'mean':
            return torch.mean(tokens, dim=1)
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

    def forward(self, instruction: BatchedInstruction):
        match self._mode:
            case 'normal':
                return self.normal_forward(instruction)
            case 'mnemonic':
                return self.mnemonic_forward(instruction)
            case 'mean':
                return self.normal_forward(instruction)
            case _:
                raise ValueError(f'Unknown mode: {self._mode}')

    def mnemonic_forward(self, instruction: BatchedInstruction):
        return self._token_embedding(instruction.mnemic)

    def normal_forward(self, instruction: BatchedInstruction):
        # 1. generate embedding for Memory Operand, token operand and immediate operand
        operands = []
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
        if instruction.has_reg_list_operand:
            reg_list = self._arm_reg_list_embedding(instruction)
            operands.append(reg_list)
        if instruction.has_reg_index_operand:
            reg_index = self._arm_reg_index_embedding(instruction)
            operands.append(reg_index)
        operands = torch.cat(operands, dim=0)
        # 2. generate embedding for mnemonic
        mnemonic_embeddings = self._token_embedding(instruction.mnemic)
        # 3. generate embedding for instruction
        instructions_index = instruction.instructions_index
        operands_index = instruction.operand_index
        for key in instructions_index:
            operand_to_ins_index = instructions_index[key]
            unique_operands_index, expand_operands_index = operands_index[key]
            if operand_to_ins_index.shape[0] == 0:
                break
            # calculate weight matrix for operands
            unique_operands_embeddings = operands[unique_operands_index]
            if self._mode == 'normal':
                unique_operands_embeddings = self._operand_linear[key](unique_operands_embeddings)
            operands_embeddings = unique_operands_embeddings[expand_operands_index]
            # calculate instruction embedding
            mnemonic_embeddings.index_add_(0, operand_to_ins_index, operands_embeddings)
        if self._mode == 'normal':
            return self._instruction_embedding_aggregator(mnemonic_embeddings)
        elif self._mode == 'mean':
            cnt = torch.zeros([mnemonic_embeddings.shape[0], 1], device=mnemonic_embeddings.device,
                                dtype=mnemonic_embeddings.dtype)
            for key in instructions_index:
                if len(instructions_index[key]) == 0:
                    break
                ones = torch.ones([instructions_index[key].shape[0], 1], device=mnemonic_embeddings.device,
                                    dtype=mnemonic_embeddings.dtype)
                cnt.index_add_(0, instructions_index[key], ones)
            return mnemonic_embeddings / (cnt + 1)

    def _mem_embedding(self, instruction: BatchedInstruction) -> List[torch.Tensor]:
        result = []
        # x86
        x86_operands = instruction.x86_mem_operands
        if x86_operands.operand_num:
            base, index, scale, offset = x86_operands.base, x86_operands.index, x86_operands.scale, x86_operands.disp
            base_index = torch.stack([base, index], dim=1)
            imm = torch.stack([scale, offset], dim=1).reshape([-1, 2, 1])
            base_index = self._token_embedding(base_index)
            imm = self._imm_embedding(imm)
            x86_mem_embedding = torch.cat([base_index, imm], dim=1)
            if self._mode == 'normal':
                result.append(self._x86_mem_embedding_aggregator(x86_mem_embedding.reshape([-1, 4 * self._embed_dim])))
            elif self._mode == 'mean':
                result.append(torch.mean(x86_mem_embedding, dim=1))
            else:
                raise ValueError(f'Unknown mode: {self._mode}')

        # arm
        arm_operands = instruction.arm_mem_operands
        if arm_operands.operand_num:
            base, index, shift_type, shift_value, offset = arm_operands.base, arm_operands.index, arm_operands.shift_type, arm_operands.shift_value, arm_operands.disp
            tokens = torch.stack([base, index, shift_type, shift_value], dim=1)
            tokens = self._token_embedding(tokens)
            imm = self._imm_embedding(offset.reshape([-1, 1, 1]))
            arm_mem_embedding = torch.cat([tokens, imm], dim=1)
            if self._mode == 'normal':
                result.append(self._arm_mem_embedding_aggregator(arm_mem_embedding.reshape([-1, 5 * self._embed_dim])))
            elif self._mode == 'mean':
                result.append(torch.mean(arm_mem_embedding, dim=1))
            else:
                raise ValueError(f'Unknown mode: {self._mode}')

        # mips
        mips_operands = instruction.mips_mem_operands
        if mips_operands.operand_num:
            base, index, offset = mips_operands.base, mips_operands.index, mips_operands.disp
            tokens = torch.stack([base, index], dim=1)
            tokens = self._token_embedding(tokens)
            imm = self._imm_embedding(offset.reshape([-1, 1, 1]))
            mips_mem_embedding = torch.cat([tokens, imm], dim=1)
            if self._mode == 'normal':
                result.append(self._mips_mem_embedding_aggregator(mips_mem_embedding.reshape([-1, 3 * self._embed_dim])))
            elif self._mode == 'mean':
                result.append(torch.mean(mips_mem_embedding, dim=1))
            else:
                raise ValueError(f'Unknown mode: {self._mode}')

        return result

    @property
    def token_embedding(self):
        return self._token_embedding
