from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module, Linear, Dropout, LayerNorm, Embedding
from .multi_head_attention import MultiHeadAttentionVarLen, MultiHeadAttentionDiGraph
from typing import Optional

class GraphTransformerEncoderLayer(Module):
    def __init__(self, d_model: int,
                 n_head: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = False,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiHeadAttentionDiGraph(d_model, n_head, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, data, rel_distance, mask, query_pos_embedding, key_pos_embedding, value_pos_embedding):
        x = data
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), rel_distance, mask, query_pos_embedding, key_pos_embedding, value_pos_embedding)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, rel_distance, mask, query_pos_embedding, key_pos_embedding, value_pos_embedding))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, rel_distance, mask, query_pos_embedding, key_pos_embedding, value_pos_embedding) -> Tensor:
        x = self.self_attn(x, rel_distance, mask, query_pos_embedding, key_pos_embedding, value_pos_embedding)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoderLayerVarLen(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = False,
                 device=None,
                 dtype=None) -> None:
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
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiHeadAttentionVarLen(d_model,
                                            n_head,
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
            cu_seqlen: Tensor,
            max_len: Tensor) -> Tensor:
        """
        Pass the input through the encoder layer.
        :param src: the sequence to the encoder layer (required).
        :param cu_seqlen: The cumulative sequence lengths of the sequences in the batch, used to index into qkv.
        :param max_len: The maximum sequence length in the batch.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), cu_seqlen, max_len)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, cu_seqlen, max_len))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  cu_seqlen: Optional[Tensor],
                  max_seqlen: Tensor,
                  ) -> Tensor:
        x = self.self_attn(x, cu_seqlen, max_seqlen)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)
