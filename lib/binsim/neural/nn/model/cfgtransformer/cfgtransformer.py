import torch
from torch import nn
import math
from binsim.neural.utils.data.dataset.pdg import BatchedInstructionSeq, BatchedFunctionSeq
from ...layer import Dense, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
class PositionalEmbedding(nn.Module):
    def __init__(self,
                 d_model,
                 max_len=1024,
                 device=None,
                 dtype=None):
        """
        The position embedding for instructions in each basic block. This implementation is copied from BERT directly.
        It should be noticed, the position index of each instruction is relative to the beginning of its basic block,
            instead of the function.
        :param d_model: The dimensional size of embedding vectors.
        :param max_len: The maximum length of our position embedding model.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # initialize the embedding model
        pe = torch.zeros(max_len, d_model, **factory_kwargs).float()
        pe.require_grad = False
        # calculate the position embedding for each position.
        position = torch.arange(0, max_len, **factory_kwargs).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.embedding = nn.Embedding.from_pretrained(pe, freeze=True, **factory_kwargs)
    def forward(self, index: torch.Tensor) -> torch.Tensor:
        """
        Just a simple forward process.
        :param index: A Tensor of [batch_size, sequence_length], the batched position index sequences of instruction sequences.
        :return: The position embedding for each instruction.
        """
        assert len(index.shape) == 2
        return self.embedding(index)


class InstructionEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden,
                 device=None,
                 dtype=None):
        """
        This is the implementation for instruction embedding. This process can be viewed as 3 steps:
        1. token --> token embedding
            In this step, each instruction is first tokenized into tokens. Then for different token types, we utilize
            different method to convert them into vectors:
            - for imm tokens, we utilize a neural network to generate its embedding;
            - for other tokens, including regs and mnemic, we just get their embedding with the embedding model.
        2. token embedding --> operand embedding
            In this step, we utilize the token embeddings to generate operand embeddings. There are usually three kinds of
            operands in assembly languages, namely immediate number, register and memory operand.
            - For immediate number and register operands, we use their token embedding as their embedding.
            - For memory operands, we define neural network for each architecture to aggregate the embedding of its tokens
                into a vector.
        3. operand embedding + mnemic embedding --> instruction
            In this step, we utilize a neural network to aggregate the mnemic embedding and operands embedding to the
            instruction embedding. As different architectures have different instruction format, we define different
            neural networks for different architectures.

        :param vocab_size: The number of different tokens in our dictionary.
        :param hidden: The dimensional size of embedding vectors.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.vocab_size = vocab_size
        self.hidden = hidden
        # The embedding model for tokens.
        self.embedding = nn.Embedding(vocab_size, hidden, **factory_kwargs)
        # The neural network for operands and instructions
        self.op_map = nn.ModuleDict({
            'op_imm': Dense(1, hidden,layers=2, **factory_kwargs),
            'op_reg': Dense(hidden, hidden,layers=2, **factory_kwargs),
            'op_x86_mem': Dense(hidden*4, hidden,layers=2, **factory_kwargs),
            'op_arm_mem': Dense(hidden*5, hidden, **factory_kwargs),
            'op_mips_mem': Dense(hidden*3, hidden, **factory_kwargs),
            'ins_x86': Dense(hidden*5, hidden, **factory_kwargs),
            'ins_arm': Dense(hidden*5, hidden, **factory_kwargs),
            'ins_mips': Dense(hidden*4, hidden, **factory_kwargs),
        })

    def _generate_mem_op_embeddings(self, imm_embedding, reg_embedding, instructions):
        if instructions.arch == 'x86':
            embeddings = [ embedding_layer[index] for embedding_layer, index in
                           zip([reg_embedding, reg_embedding, imm_embedding, imm_embedding], instructions.mem_op_idx)]
            mem_embedding = torch.cat(embeddings, dim=-1)
        elif instructions.arch == 'arm':
            embeddings = [ embedding_layer[index] for embedding_layer, index in
                           zip([reg_embedding, reg_embedding, imm_embedding, imm_embedding, reg_embedding],
                               instructions.mem_op_idx)]
            mem_embedding = torch.cat(embeddings, dim=-1)
        elif instructions.arch == 'mips':
            embeddings = [ embedding_layer[index] for embedding_layer, index in
                           zip([reg_embedding, reg_embedding, imm_embedding], instructions.mem_op_idx)]
            mem_embedding = torch.cat(embeddings, dim=-1)
        else:
            raise NotImplementedError(f"Unsupported architecture: {instructions.arch}")
        return self.op_map[f'op_{instructions.arch}_mem'](mem_embedding)

    def forward(self, instructions:BatchedInstructionSeq)->torch.Tensor:
        """
        Generate embedding for the instructions.
        :param instructions:
        :return:
        """
        # generate embeddings for tokens
        imm, regs = torch.tanh(instructions.imm), instructions.regs
        regs = self.embedding(regs)
        # generate embeddings for register operands and immediate number operands
        op_reg = self.op_map['op_reg'](regs)
        op_imm = self.op_map['op_imm'](imm)
        op_mem = self._generate_mem_op_embeddings(op_imm, regs, instructions)
        # generate embeddings for instructions
        operands = torch.cat([op_reg, op_imm, op_mem], dim=0)
        mnemic = self.embedding(instructions.mnemic)
        operand_idx = instructions.operand_idx
        mnemic_idx = instructions.mnemic_idx
        ins_embedding = torch.cat([mnemic[mnemic_idx],
                                   torch.reshape(operands[operand_idx], [mnemic_idx.shape[0],mnemic_idx.shape[1], -1])],
                                  dim=-1)
        ins_embedding = self.op_map[f'ins_{instructions.arch}'](ins_embedding)
        return ins_embedding

class CFGTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden=128,
                 n_layers=12,
                 n_heads=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 dtype=None,
                 device=None,
                 max_relative_position=-1,
                 checkpoint_segments=0):
        """

        :param vocab_size: The number of different tokens in our dictionary.
        :param hidden:  The dimensional size of embedding vectors.
        :param n_layers: The number of layers in the transformer encoder.
        :param dropout:
        :param n_heads: The number of heads.
        """
        super().__init__()
        factory_kwargs = {'dtype':dtype, 'device': device}
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = n_heads
        self.dropout = dropout
        self.max_relative_position = max_relative_position
        self._checkpoint_segments = checkpoint_segments
        # absolute position embedding layer
        if max_relative_position <= 0:
            self.positional_embedding = PositionalEmbedding(hidden, **factory_kwargs)
        else:
            self.positional_embedding = None
        # instruction embedding layer
        self.ins_embedding = InstructionEmbedding(vocab_size, hidden, **factory_kwargs)
        # transformer encoder layers
        self.layers = torch.nn.ModuleList([TransformerEncoderLayer(
            hidden,
            self.attn_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_relative_position = max_relative_position,
            **factory_kwargs)
            for _ in range(self.n_layers)])

    def _checkpoint_forward_segment(self,
                            start,
                            end,
                            ins_embedding,
                            src_key_padding_mask=None,
                            relative_position_idx=None):
        for layer in self.layers[start:end]:
            ins_embedding = layer(ins_embedding,
                                  src_key_padding_mask=src_key_padding_mask,
                                  relative_position_idx=relative_position_idx)
        return ins_embedding

    def _checkpoint_forward(self,
                            ins_embedding:torch.Tensor,
                            src_key_padding_mask=None,
                            relative_position_idx=None):
        layers_per_segment = self.n_layers // self._checkpoint_segments
        for start in range(0, self.n_layers, layers_per_segment):
            end = start + layers_per_segment
            ins_embedding = checkpoint(self._checkpoint_forward_segment,
                                       start,
                                       end,
                                       ins_embedding,
                                       src_key_padding_mask,
                                       relative_position_idx,
                                       preserve_rng_state=True)
        return ins_embedding


    def forward(self, instructions: BatchedFunctionSeq)->torch.Tensor:
        ins_embedding = self.ins_embedding(instructions)
        relative_position_offset = instructions.relative_position_offset

        flow_mask = torch.isinf(relative_position_offset)
        relative_position_offset = torch.clamp_(relative_position_offset, -self.max_relative_position, self.max_relative_position)
        if self._checkpoint_segments > 1:
            ins_embedding = checkpoint(self._checkpoint_forward,
                                       ins_embedding,
                                       flow_mask,
                                       relative_position_offset)
        else:
            for transformer in self.layers:
                ins_embedding = transformer(ins_embedding,
                                            src_key_padding_mask=flow_mask,
                                            relative_position_idx=relative_position_offset)
        return ins_embedding
