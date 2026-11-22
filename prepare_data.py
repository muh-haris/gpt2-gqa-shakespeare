from pathlib import Path
from urllib.request import urlopen

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

PROJECT_DIR = Path(__file__).resolve().parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "shakespeare.txt"
TRAIN_FILE = PROJECT_DIR / "data" / "processed" / "train.txt"
VAL_FILE = PROJECT_DIR / "data" / "processed" / "val.txt"


def split_text(text, train_fraction=0.9):
    """Split text in order without mixing the two sections."""
    if len(text) < 2 or not 0 < train_fraction < 1:
        raise ValueError("Invalid dataset or split fraction")

    split_at = int(len(text) * train_fraction)

    if split_at == 0 or split_at == len(text):
        raise ValueError("Split would produce an empty section")

    return text[:split_at], text[split_at:]


def prepare_data():
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        print("Downloading Tiny Shakespeare...")

        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()

        RAW_FILE.write_bytes(content)

    text = RAW_FILE.read_text(encoding="utf-8")

    train_text, val_text = split_text(text)

    with TRAIN_FILE.open("w", encoding="utf-8", newline="\n") as f:
        f.write(train_text)

    with VAL_FILE.open("w", encoding="utf-8", newline="\n") as f:
        f.write(val_text)

    print(f"Total characters: {len(text):,}")
    print(f"Training characters: {len(train_text):,}")
    print(f"Validation characters: {len(val_text):,}")
    print(f"Unique characters: {len(set(text))}")
    print("Dataset preparation completed.")


if __name__ == "__main__":
    prepare_data()
