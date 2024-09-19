import torch
import math
from torch import nn

class MultiHeadAttentionVarLen(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 device=None,
                 dtype=None):
        super().__init__()
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_qkv = nn.Linear(hid_dim, hid_dim*3, **factory_kwargs)
        self.scale = torch.tensor(math.sqrt(self.head_dim),**factory_kwargs)

    def forward(self, x, cu_seq_len:torch.Tensor, max_seqlen:int):
        """
        The forward process of MultiHeadAttention.
        :param x: (total, hidden_dim), dtype torch.float32. The packed token sequence.
        :param cu_seq_len: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        :param max_seqlen: int. Maximum sequence length in the batch.
        :return:
        """
        from flash_attn import flash_attn_varlen_qkvpacked_func
        token_num = x.size(0)
        qkv = self.fc_qkv(x).view(token_num, 3, self.n_heads, self.head_dim)
        out = flash_attn_varlen_qkvpacked_func(qkv, cu_seq_len, max_seqlen, softmax_scale=self.scale)
        return out.resize(token_num, self.hid_dim)

class MultiHeadAttentionDiGraph(nn.Module):
    def __init__(self, hidden_dim, num_heads, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.fc_q = nn.Linear(hidden_dim, hidden_dim, bias=False, **factory_kwargs)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim * 2, bias=False, **factory_kwargs)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim * 2, bias=False, **factory_kwargs)
        self.scale = torch.tensor(math.sqrt(self.head_dim), **factory_kwargs)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, **factory_kwargs)

    def calculate_rel_distance_weight(self, query, dis_embed, rel_distance):
        batch_size, _, query_seq_len, key_seq_len = rel_distance.shape
        q_dis_embed = dis_embed.view(1, dis_embed.shape[0], self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        query_pos_weight = torch.matmul(query, q_dis_embed)  # (batch_size, num_heads, query_seq_len, dist_num)
        query_pos_weights = torch.gather(query_pos_weight, 3, rel_distance)
        return query_pos_weights

    def forward(self, x, rel_distance, mask, query_pos_embedding, key_pos_embedding, value_pos_embedding):
        batch_size, seq_len, _ = x.size()
        dis_num = value_pos_embedding.shape[0]
        q = self.fc_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.fc_k(x).view(batch_size, seq_len * 2, self.num_heads, self.head_dim)
        v = self.fc_v(x).view(batch_size, seq_len * 2, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q, k, v: (batch_size, num_heads, seq_len, head_dim)

        rel_distance = rel_distance.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        query_pos_weights = self.calculate_rel_distance_weight(q, query_pos_embedding, rel_distance)
        key_pos_weights = self.calculate_rel_distance_weight(k, key_pos_embedding, rel_distance.transpose(2, 3)).transpose(2, 3)


        x = torch.matmul(q, k.transpose(-2, -1))
        x = (x + query_pos_weights + key_pos_weights) / self.scale

        if mask is not None:
            x = x.masked_fill(mask.view(batch_size, 1, seq_len, seq_len * 2), float('-inf'))

        x = torch.softmax(x, dim=-1)

        # v_dis_embed_fwd: (batch_size, dist_num, hidden_dim)
        # v_dis_embed_bck: (batch_size, dist_num, hidden_dim)
        value_pos_embedding = value_pos_embedding.view(1, dis_num, self.num_heads, self.head_dim).transpose(1, 2)

        v_dis_embed_weights = torch.zeros((batch_size, self.num_heads, seq_len, dis_num), device=x.device)
        v_dis_embed_weights = torch.scatter_add(v_dis_embed_weights, 3, rel_distance, x)

        x = x.matmul(v) + torch.matmul(v_dis_embed_weights, value_pos_embedding)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        x = self.out_proj(x)
        return x



