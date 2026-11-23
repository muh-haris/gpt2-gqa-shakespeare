
import hashlib
from pathlib import Path

import torch

from src.config import GPTConfig
from src.generation import generate_tokens
from src.model import GPTLanguageModel
from src.tokenizer import BPETokenizer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "checkpoints" / "gpt_gqa_best.pt"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = {
        "train.txt": DATA / "train.txt",
        "val.txt": DATA / "val.txt",
        "tokenizer.json": DATA / "tokenizer.json",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError("Best model checkpoint is missing")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    for name, path in files.items():
        expected = checkpoint["fingerprints"].get(name)

        if expected != sha256(path):
            raise ValueError(f"Fingerprint mismatch: {name}")

    tokenizer = BPETokenizer.load(files["tokenizer.json"])
    config = GPTConfig(**checkpoint["config"])

    assert tokenizer.vocab_size == config.vocab_size
    assert config.n_heads == 8
    assert config.n_kv_heads == 2

    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt = "ROMEO:"
    prompt_ids = tokenizer.encode(prompt)

    output_ids = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=20,
        temperature=0.9,
        top_k=40,
        seed=42,
    )

    assert len(output_ids) == len(prompt_ids) + 20
    assert output_ids[:len(prompt_ids)] == prompt_ids

    generated = tokenizer.decode(output_ids, errors="replace")

    print("Dataset fingerprints: Verified")
    print("Tokenizer: Verified")
    print("Checkpoint: Loaded")
    print("Architecture: GPT-2-style GQA")
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
    print("Training step:", checkpoint["step"])
    print("Validation loss:", checkpoint["val_loss"])
    print("Generated new tokens: 20")
    print("Sample:", repr(generated))
    print("End-to-end verification: PASSED")


if __name__ == "__main__":
    main()
