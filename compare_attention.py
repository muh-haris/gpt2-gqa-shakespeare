
import torch

from src.config import GPTConfig
from src.model import GPTLanguageModel


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def main():
    torch.manual_seed(42)

    mha = GPTLanguageModel(GPTConfig(n_kv_heads=8))
    gqa = GPTLanguageModel(GPTConfig(n_kv_heads=2))

    mha_total = count_parameters(mha)
    gqa_total = count_parameters(gqa)

    mha_attention = count_parameters(mha.blocks[0].attn)
    gqa_attention = count_parameters(gqa.blocks[0].attn)

    print("MHA total parameters:", f"{mha_total:,}")
    print("GQA total parameters:", f"{gqa_total:,}")
    print("Total parameters saved:", f"{mha_total - gqa_total:,}")

    print("MHA attention parameters:", f"{mha_attention:,}")
    print("GQA attention parameters:", f"{gqa_attention:,}")

    reduction = (1 - gqa_total / mha_total) * 100
    print(f"Overall parameter reduction: {reduction:.2f}%")

    ids = torch.randint(0, 512, (2, 16))
    gqa.eval()

    with torch.no_grad():
        logits, _ = gqa(ids)

    print("GQA output shape:", tuple(logits.shape))
    print("Finite logits:", torch.isfinite(logits).all().item())


if __name__ == "__main__":
    main()
