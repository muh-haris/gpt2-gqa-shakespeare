
import argparse
import csv
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.tokenizer import BPETokenizer
from src.training import evaluate, learning_rate, sample_batch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "checkpoints" / "gpt_gqa_latest.pt"
METRICS = ROOT / "runs" / "training_log.csv"
BEST_CHECKPOINT = ROOT / "checkpoints" / "gpt_gqa_best.pt"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(model, optimizer, sampler, step, config, fingerprints):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "config": asdict(config),
        "fingerprints": fingerprints,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "sampler_rng": sampler.get_state(),
    }

    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()

    temporary = CHECKPOINT.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(CHECKPOINT)


def read_checkpoint(model, optimizer, sampler, config, fingerprints, device):
    state = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=True
    )

    if state["config"] != asdict(config):
        raise ValueError("Checkpoint architecture does not match the current model")

    if state["fingerprints"] != fingerprints:
        raise ValueError("Tokenizer or data changed since checkpoint creation")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    torch.set_rng_state(state["torch_rng"])
    sampler.set_state(state["sampler_rng"])

    if device.type == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    return int(state["step"])


def record(step, train_loss, val_loss, lr, new_run=False):
    METRICS.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if new_run else "a"
    needs_header = new_run or not METRICS.exists()

    with METRICS.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)

        if needs_header:
            writer.writerow([
                "step", "train_loss", "val_loss", "learning_rate"
            ])

        writer.writerow([step, train_loss, val_loss, lr])


def save_best_model(model, step, val_loss, config, fingerprints):
    BEST_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "val_loss": float(val_loss),
        "config": asdict(config),
        "fingerprints": fingerprints,
        "model": model.state_dict(),
    }

    temporary = BEST_CHECKPOINT.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(BEST_CHECKPOINT)


def run(args):
    if args.steps <= 0 or args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("Steps, batch size and sequence length must be positive")

    if min(
        args.eval_batches, args.eval_every,
        args.log_every, args.save_every
    ) <= 0:
        raise ValueError("Evaluation, logging and saving intervals must be positive")

    if args.grad_clip <= 0 or args.threads < 0:
        raise ValueError("Invalid clipping or CPU thread setting")

    if args.threads:
        torch.set_num_threads(args.threads)

    learning_rate(
        args.steps, args.steps, args.warmup_steps,
        args.lr, args.min_lr
    )

    required = [
        DATA / "train.txt",
        DATA / "val.txt",
        DATA / "tokenizer.json",
    ]

    for file in required:
        if not file.exists():
            raise FileNotFoundError(
                f"Missing {file.name}. Prepare the dataset and tokenizer first."
            )

    fingerprints = {p.name: digest(p) for p in required}

    tokenizer = BPETokenizer.load(DATA / "tokenizer.json")

    train_ids = torch.tensor(
        tokenizer.encode(
            (DATA / "train.txt").read_text(encoding="utf-8")
        ),
        dtype=torch.long,
    )

    val_ids = torch.tensor(
        tokenizer.encode(
            (DATA / "val.txt").read_text(encoding="utf-8")
        ),
        dtype=torch.long,
    )

    config = GPTConfig(vocab_size=tokenizer.vocab_size)

    if args.seq_len > config.context_length:
        raise ValueError(
            f"Sequence length exceeds model context length {config.context_length}"
        )

    if min(len(train_ids), len(val_ids)) <= args.seq_len:
        raise ValueError("Tokenized dataset is too short")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = GPTLanguageModel(config).to(device)

    decay = [
        p for p in model.parameters()
        if p.requires_grad and p.ndim >= 2
    ]
    no_decay = [
        p for p in model.parameters()
        if p.requires_grad and p.ndim < 2
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    sampler = torch.Generator().manual_seed(args.seed + 1)

    start = 0
    best_val = float("inf")

    if args.resume:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {CHECKPOINT}"
            )

        start = read_checkpoint(
            model, optimizer, sampler, config, fingerprints, device
        )

        if BEST_CHECKPOINT.exists():
            previous_best = torch.load(
                BEST_CHECKPOINT,
                map_location="cpu",
                weights_only=True,
            )

            if (
                previous_best["config"] != asdict(config)
                or previous_best["fingerprints"] != fingerprints
            ):
                raise ValueError("Best checkpoint belongs to another run")

            best_val = float(previous_best["val_loss"])

        if args.steps <= start:
            raise ValueError(
                f"--steps must be greater than checkpoint step {start}"
            )

        print(f"Resumed from step {start}")
    else:
        BEST_CHECKPOINT.unlink(missing_ok=True)

        initial_val = evaluate(
            model, val_ids, args.seq_len,
            args.batch_size, args.eval_batches, device
        )

        record(
            0, "", f"{initial_val:.4f}", 0.0, new_run=True
        )
        print(f"Step 0 | val loss {initial_val:.4f}")

    print(
        f"Device: {device}; "
        f"train tokens: {len(train_ids):,}; "
        f"validation tokens: {len(val_ids):,}"
    )

    for step in range(start + 1, args.steps + 1):
        model.train()

        lr = learning_rate(
            step, args.steps, args.warmup_steps,
            args.lr, args.min_lr
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = sample_batch(
            train_ids, args.batch_size,
            args.seq_len, device, sampler
        )

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )

        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            print(
                f"Step {step}/{args.steps} | "
                f"train loss {loss.item():.4f} | "
                f"lr {lr:.6g}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(
                model, val_ids, args.seq_len,
                args.batch_size, args.eval_batches, device
            )

            record(
                step, f"{loss.item():.4f}",
                f"{val_loss:.4f}", lr
            )

            print(f"              val loss {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                save_best_model(
                    model, step, val_loss, config, fingerprints
                )
                print(f"              new best loss {best_val:.4f}")

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(
                model, optimizer, sampler,
                step, config, fingerprints
            )
            print(f"Saved checkpoint at step {step}")

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Metrics: {METRICS}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Shakespeare GPT with GQA"
    )

    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    run(parser.parse_args())


if __name__ == "__main__":
    main()
