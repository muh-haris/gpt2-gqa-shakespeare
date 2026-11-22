import torch

from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.tokenizer import BPETokenizer


def main():
    torch.manual_seed(42)

    tokenizer = BPETokenizer.load(
        "data/processed/tokenizer.json"
    )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size
    )

    model = GPTLanguageModel(config)

    text = (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n"
    )

    ids = tokenizer.encode(text)

    if len(ids) < 2:
        raise ValueError("Need at least two tokens")

    count = min(len(ids) - 1, config.context_length)

    x = torch.tensor(
        [ids[:count]], dtype=torch.long
    )
    y = torch.tensor(
        [ids[1:count + 1]], dtype=torch.long
    )

    model.eval()

    with torch.no_grad():
        logits, loss = model(x, y)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", f"{parameters:,}")
    print("Logits shape:", tuple(logits.shape))
    print("Initial loss:", round(loss.item(), 4))
    print("Model has not been trained yet.")


if __name__ == "__main__":
    main()
