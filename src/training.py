
import math

import torch


def sample_batch(token_ids, batch_size, seq_len, device, generator):
    if token_ids.ndim != 1 or len(token_ids) <= seq_len:
        raise ValueError("Not enough tokens for the chosen sequence length")

    starts = torch.randint(
        0, len(token_ids) - seq_len,
        (batch_size,), generator=generator
    )

    positions = starts[:, None] + torch.arange(seq_len)[None, :]

    x = token_ids[positions]
    y = token_ids[positions + 1]

    return x.to(device), y.to(device)


def learning_rate(step, total_steps, warmup_steps, max_lr, min_lr):
    if not (0 <= warmup_steps < total_steps):
        raise ValueError("Warmup steps must be less than total training steps")

    if not (0 < min_lr <= max_lr):
        raise ValueError("Learning rates must be positive and min_lr <= max_lr")

    if warmup_steps and step <= warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    return min_lr + 0.5 * (max_lr - min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def evaluate(model, val_ids, seq_len, batch_size, n_batches, device):
    was_training = model.training
    model.eval()

    # Use identical validation windows at each evaluation.
    sampler = torch.Generator().manual_seed(2026)
    losses = []

    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = sample_batch(
                    val_ids, batch_size, seq_len, device, sampler
                )
                _, loss = model(x, y)
                losses.append(loss.item())
    finally:
        model.train(was_training)

    return sum(losses) / len(losses)
