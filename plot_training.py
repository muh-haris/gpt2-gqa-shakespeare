
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "runs" / "training_log.csv"
FIGURE_FILE = ROOT / "docs" / "results" / "gpt07_loss_curve.png"


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError("Training log not found")

    with LOG_FILE.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    train_steps = [
        int(row["step"]) for row in rows if row["train_loss"]
    ]
    train_losses = [
        float(row["train_loss"]) for row in rows if row["train_loss"]
    ]

    val_steps = [
        int(row["step"]) for row in rows if row["val_loss"]
    ]
    val_losses = [
        float(row["val_loss"]) for row in rows if row["val_loss"]
    ]

    if not val_losses:
        raise ValueError("No validation metrics were recorded")

    best_index = min(
        range(1, len(val_losses)),
        key=lambda i: val_losses[i],
    ) if len(val_losses) > 1 else 0

    plt.figure(figsize=(9, 5))
    plt.plot(
        train_steps, train_losses,
        marker="o", label="Training minibatch loss"
    )
    plt.plot(
        val_steps, val_losses,
        marker="s", label="Validation loss"
    )

    plt.scatter(
        val_steps[best_index],
        val_losses[best_index],
        marker="*",
        s=150,
        label="Lowest logged validation loss",
    )

    plt.xlabel("Optimizer step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Shakespeare GPT-2 with GQA")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_FILE, dpi=160)
    plt.close()

    print("Figure saved:", FIGURE_FILE)
    print("Initial validation loss:", val_losses[0])
    print("Best logged validation step:", val_steps[best_index])
    print("Best logged validation loss:", val_losses[best_index])


if __name__ == "__main__":
    main()
