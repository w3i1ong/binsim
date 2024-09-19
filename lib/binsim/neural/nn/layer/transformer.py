import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import Module, Linear, Dropout, LayerNorm
from .multi_head_attention import MultiHeadAttention
from typing import Callable, Union, Optional

class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = False,
                 device=None,
                 dtype=None,
                 max_relative_position=-1) -> None:
        """
        This is a modified version of the original TransformerEncoderLayer in PyTorch. The original version only supports
        absolute positional embedding. I modified it to support relative positional embedding.
        :param d_model: The number of expected features in the input (required).
        :param n_head: The number of heads in the MultiHeadAttention models (required).
        :param dim_feedforward: The dimension of the feedforward network model (default=2048).
        :param dropout: The dropout value (default=0.1).
        :param layer_norm_eps: The epsilon value to use in :class:`torch.nn.LayerNorm` (default=1e-5).
        :param norm_first: If ``True``, then the normalization layer is put before the self-attention layer.
        :param device: The device of the model.
        :param dtype: The dtype of the model.
        :param max_relative_position: The maximum relative position. If it is -1, then relative position is not used.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiHeadAttention(d_model,
                                            n_head,
                                            dropout=dropout,
                                            max_relative_position=max_relative_position,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(
            self,
            src: Tensor,
            src_key_padding_mask: Optional[Tensor] = None,
            relative_position_idx: Tensor=None) -> Tensor:
        """
        Pass the input through the encoder layer.
        :param src: the sequence to the encoder layer (required).
        :param src_key_padding_mask: the mask between query matrix and key matrix.
        :param relative_position_idx: the relative distance between each element pair.
        """
        # the dtype must be boolean
        assert src_key_padding_mask.dtype == torch.bool, "The dtype of src_key_padding_mask must" \
                                                         "be torch.bool!"

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_key_padding_mask, relative_position_idx)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_key_padding_mask, relative_position_idx))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  src_key_padding_mask: Optional[Tensor],
                  relative_position_idx: Tensor,
                  ) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=src_key_padding_mask,
                           relative_distance=relative_position_idx)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)
