
import pytest
import torch

import train
from src.config import GPTConfig
from src.model import GPTLanguageModel
from src.training import evaluate, learning_rate, sample_batch


def test_targets_are_shifted():
    tokens = torch.arange(100)
    generator = torch.Generator().manual_seed(4)

    x, y = sample_batch(tokens, 3, 8, "cpu", generator)

    assert x.shape == y.shape == (3, 8)
    assert torch.equal(y, x + 1)


def test_batch_reproducibility():
    tokens = torch.arange(100)

    a = torch.Generator().manual_seed(3)
    b = torch.Generator().manual_seed(3)

    x1, y1 = sample_batch(tokens, 2, 5, "cpu", a)
    x2, y2 = sample_batch(tokens, 2, 5, "cpu", b)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_short_dataset_rejected():
    with pytest.raises(ValueError):
        sample_batch(
            torch.arange(8), 2, 8, "cpu", torch.Generator()
        )


def test_learning_rate_schedule():
    assert learning_rate(
        2, 10, 2, 0.001, 0.0001
    ) == pytest.approx(0.001)

    assert learning_rate(
        10, 10, 2, 0.001, 0.0001
    ) == pytest.approx(0.0001)

    assert learning_rate(
        1, 10, 2, 0.001, 0.0001
    ) == pytest.approx(0.0005)


def test_evaluation_is_repeatable():
    torch.manual_seed(4)

    config = GPTConfig(
        vocab_size=512,
        context_length=16,
        n_embd=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.1,
    )

    model = GPTLanguageModel(config)
    ids = torch.randint(0, 512, (60,))

    first = evaluate(model, ids, 8, 2, 2, "cpu")
    second = evaluate(model, ids, 8, 2, 2, "cpu")

    assert first == pytest.approx(second)
    assert model.training


def test_checkpoint_restores_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train, "CHECKPOINT", tmp_path / "model.pt"
    )

    config = GPTConfig(
        vocab_size=512,
        context_length=16,
        n_embd=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
    )

    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001
    )

    sampler = torch.Generator().manual_seed(20)
    fingerprints = {"train.txt": "test"}

    before = model.token_embedding.weight.detach().clone()

    train.save_checkpoint(
        model, optimizer, sampler,
        3, config, fingerprints
    )

    with torch.no_grad():
        model.token_embedding.weight.add_(2.0)

    restored_step = train.read_checkpoint(
        model, optimizer, sampler,
        config, fingerprints, torch.device("cpu")
    )

    assert restored_step == 3
    assert torch.equal(
        model.token_embedding.weight, before
    )
