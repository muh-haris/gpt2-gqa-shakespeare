
import torch
from torch.nn import functional as F


def generate_tokens(
    model,
    prompt_ids,
    max_new_tokens=100,
    temperature=0.9,
    top_k=40,
    seed=42,
):
    if not prompt_ids:
        raise ValueError("A non-empty prompt is required")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    if any(
        not isinstance(token, int)
        or not 0 <= token < model.config.vocab_size
        for token in prompt_ids
    ):
        raise ValueError("Prompt contains invalid token IDs")

    device = next(model.parameters()).device

    ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )

    sampler = torch.Generator(device=device).manual_seed(seed)

    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = ids[:, -model.config.context_length:]

                logits, _ = model(context)
                last_logits = logits[:, -1, :] / temperature

                if top_k is None:
                    probabilities = F.softmax(last_logits, dim=-1)

                    next_id = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=sampler,
                    )
                else:
                    scores, candidate_ids = torch.topk(
                        last_logits,
                        k=min(top_k, last_logits.size(-1)),
                    )

                    probabilities = F.softmax(scores, dim=-1)

                    selected = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=sampler,
                    )

                    next_id = candidate_ids.gather(1, selected)

                ids = torch.cat((ids, next_id), dim=1)

    finally:
        model.train(was_training)

    return ids[0].tolist()
