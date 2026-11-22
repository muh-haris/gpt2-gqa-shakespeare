import torch
from torch import nn
from torch.nn import functional as F

from src.attention import CausalSelfAttention
from src.config import GPTConfig


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_attn = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.n_embd
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        # Input and output share the same token weights.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Smaller initialization for residual projections.
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=0.02 / (2 * config.n_layers) ** 0.5,
            )
            nn.init.normal_(
                block.mlp.net[2].weight,
                mean=0.0,
                std=0.02 / (2 * config.n_layers) ** 0.5,
            )

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError(
                "Token IDs must have shape (batch, sequence)"
            )

        batch, length = idx.shape

        if length < 1 or length > self.config.context_length:
            raise ValueError(
                "Sequence length is outside the model context"
            )

        if targets is not None and targets.shape != idx.shape:
            raise ValueError(
                "Targets must match the shape of input IDs"
            )

        positions = torch.arange(length, device=idx.device)

        x = self.token_embedding(idx)
        x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss
