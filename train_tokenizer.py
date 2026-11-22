from pathlib import Path

from src.tokenizer import BPETokenizer


DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train.txt"
VAL_PATH = DATA_DIR / "val.txt"
MODEL_PATH = DATA_DIR / "tokenizer.json"


def main():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(
            "Run python prepare_data.py before training the tokenizer."
        )

    train_text = TRAIN_PATH.read_text(encoding="utf-8")
    val_text = VAL_PATH.read_text(encoding="utf-8")

    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer.train(train_text, vocab_size=512)

    tokenizer.save(MODEL_PATH)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Learned merges: {len(tokenizer.merges)}")

    for name, text in [
        ("Training", train_text),
        ("Validation", val_text)
    ]:
        ids = tokenizer.encode(text)
        restored = tokenizer.decode(ids)

        assert restored == text, f"{name} round-trip failed"

        byte_count = len(text.encode("utf-8"))
        print(f"{name} bytes: {byte_count:,}")
        print(f"{name} BPE tokens: {len(ids):,}")
        print(f"{name} bytes per token: {byte_count / len(ids):.2f}")

    example = "To be, or not to be."
    print("Example text:", example)
    print("Token IDs:", tokenizer.encode(example))
    print("Decoded:", tokenizer.decode(tokenizer.encode(example)))
    print(f"Tokenizer saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
