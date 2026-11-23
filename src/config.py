
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 512
    context_length: int = 128
    n_embd: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 1 or self.context_length < 1:
            raise ValueError("Vocabulary size and context length must be positive")

        if self.n_embd < 1 or self.n_layers < 1 or self.n_heads < 1:
            raise ValueError("Model dimensions must be positive")

        if self.n_embd % self.n_heads != 0:
            raise ValueError("Embedding size must be divisible by query heads")

        if self.n_kv_heads < 1 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("KV heads must be positive and divide query heads")

        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be between 0 and 1")
