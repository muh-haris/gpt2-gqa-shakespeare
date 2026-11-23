
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.n_embd // config.n_heads
        self.group_size = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        kv_width = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, kv_width)
        self.v_proj = nn.Linear(config.n_embd, kv_width)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def split_heads(self, x, n_heads):
        batch, length, _ = x.shape

        return x.reshape(
            batch, length, n_heads, self.head_dim
        ).transpose(1, 2)

    def expand_kv(self, x):
        if self.group_size == 1:
            return x

        return x.repeat_interleave(self.group_size, dim=1)

    def forward(self, x):
        q = self.split_heads(self.q_proj(x), self.n_heads)
        k = self.split_heads(self.k_proj(x), self.n_kv_heads)
        v = self.split_heads(self.v_proj(x), self.n_kv_heads)

        k = self.expand_kv(k)
        v = self.expand_kv(v)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        batch, _, length, _ = y.shape

        y = y.transpose(1, 2).contiguous()
        y = y.reshape(batch, length, -1)

        return self.resid_dropout(self.out_proj(y))
