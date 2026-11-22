from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def split_heads(self, x):
        batch, length, _ = x.shape

        x = x.view(
            batch, length, self.n_heads, self.head_dim
        )

        return x.transpose(1, 2)

    def forward(self, x):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        batch, _, length, _ = y.shape

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch, length, -1)

        return self.resid_dropout(self.out_proj(y))
