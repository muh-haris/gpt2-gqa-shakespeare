import pytest
import torch

from src.config import GPTConfig
from src.model import GPTLanguageModel


@pytest.fixture
def small_model():
    torch.manual_seed(7)

    config = GPTConfig(
        vocab_size=512,
        context_length=16,
        n_embd=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )

    return GPTLanguageModel(config)


def test_logits_shape(small_model):
    ids = torch.randint(0, 512, (2, 8))
    logits, loss = small_model(ids)

    assert logits.shape == (2, 8, 512)
    assert loss is None


def test_loss_and_gradients(small_model):
    ids = torch.randint(0, 512, (2, 8))
    targets = torch.randint(0, 512, (2, 8))

    _, loss = small_model(ids, targets)

    assert torch.isfinite(loss)

    loss.backward()

    assert small_model.token_embedding.weight.grad is not None
    assert small_model.blocks[0].attn.q_proj.weight.grad is not None


def test_causal_mask(small_model):
    small_model.eval()

    left = torch.tensor([[10, 20, 30, 40, 50]])
    right = torch.tensor([[10, 20, 30, 90, 100]])

    with torch.no_grad():
        a, _ = small_model(left)
        b, _ = small_model(right)

    assert torch.allclose(
        a[:, :3], b[:, :3], atol=1e-5
    )


def test_weight_tying(small_model):
    assert (
        small_model.lm_head.weight
        is small_model.token_embedding.weight
    )


def test_context_limit(small_model):
    with pytest.raises(ValueError):
        small_model(
            torch.ones((1, 17), dtype=torch.long)
        )


def test_invalid_head_count():
    with pytest.raises(ValueError):
        GPTConfig(n_embd=30, n_heads=8)
