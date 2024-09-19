import torch
import math
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 device=None,
                 dtype=None,
                 dropout=None,
                 max_relative_position=-1):
        super().__init__()
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = max_relative_position

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc_q = nn.Linear(hid_dim, hid_dim, **factory_kwargs)
        self.fc_k = nn.Linear(hid_dim, hid_dim, **factory_kwargs)
        self.fc_v = nn.Linear(hid_dim, hid_dim, **factory_kwargs)
        self.fc_o = nn.Linear(hid_dim, hid_dim, **factory_kwargs)
        self.scale = torch.tensor(math.sqrt(self.head_dim),**factory_kwargs)

        if max_relative_position > 0:
            self.relative_position_embedding = nn.Embedding(max_relative_position*2+1, self.head_dim, **factory_kwargs)
        else:
            self.relative_position_embedding = None

    def forward(self, query:torch.Tensor,
                key:torch.Tensor,
                value:torch.Tensor,
                attn_mask:torch.Tensor=None,
                relative_distance: torch.Tensor=None)->torch.Tensor:
        """
        The forward process of MultiHeadAttention.
        :param query: The query tensor. Shape of [batch size, query len, hid dim]
        :param key: The key tensor. Shape of [batch size, key len, hid dim]
        :param value: The value tensor. Shape of [batch size, value len, hid dim]
        :param attn_mask: The attention mask. Shape of [batch size, query len, key len] or [batch size, n heads, query len, key len]
        :param relative_distance: The relative distance tensor. Shape of [batch size, query len, key len] or [batch size, n heads, query len, key len]
        :return:
        """
        batch_size = query.shape[0]
        query = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # shape of query: [batch size, n heads, query len, head dim]
        key = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        # shape of key: [batch size, n heads, key len, head dim]
        value = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # shape of value: [batch size, n heads, value len, head dim]

        attn:torch.Tensor = torch.matmul(query, key)/self.scale
        # shape of attn: [batch size, n heads, query len, key len]

        # prepare relative positional embedding
        if relative_distance is not None:
            assert self.relative_position_embedding is not None, "Current model does not support relative position embedding." \
                                                                 "If you want to use relative position embedding, please" \
                                                                 " set max_relative_position > 0 in the constructor."
            if len(relative_distance.shape) != 3:
                raise ValueError("The shape of relative distance must be 3 dimensions.")
            assert relative_distance.min() >= -self.max_relative_position and relative_distance.max() <= self.max_relative_position, \
                "The relative distance must be in the range of [-max_relative_position, max_relative_position]."

            relative_distance = relative_distance + self.max_relative_position
            relative_position_embedding = self.relative_position_embedding(relative_distance)
            # shape: [batch size, query len, key len, head dim]
            relative_position_attn: torch.Tensor = torch.matmul(query.permute(0,2,1,3), relative_position_embedding.transpose(2, 3))/self.scale
            # shape: [batch size, query len, head num, key len]
            relative_position_attn = relative_position_attn.permute(0,2,1,3)
            attn = attn + relative_position_attn

        if attn_mask is not None:
            if len(attn_mask.shape) == 3:
                attn_mask = attn_mask.view(batch_size, 1, attn_mask.shape[1], attn_mask.shape[2])
            if len(attn_mask.shape) != 4:
                raise ValueError("The shape of attention mask must be 3 or 4 dimensions.")
            assert attn_mask.dtype == torch.bool, "The dtype of src_key_padding_mask must"\
                                                  "be torch.bool!"
            attn = attn.masked_fill(attn_mask, -torch.inf)
        if self.dropout is not None:
            attn = self.dropout(attn)

        attn = torch.softmax(attn, dim=-1)
        # attn = [batch size, n heads, query len, key len]
        x = torch.matmul(attn, value)
        # x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x
