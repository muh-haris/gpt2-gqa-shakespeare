
import pytest
import torch

from src.config import GPTConfig
from src.generation import generate_tokens
from src.model import GPTLanguageModel


@pytest.fixture
def model():
    torch.manual_seed(11)

    config = GPTConfig(
        vocab_size=32,
        context_length=8,
        n_embd=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
    )

    return GPTLanguageModel(config)


def test_output_length_and_prefix(model):
    prompt = [1, 2, 3]

    result = generate_tokens(
        model, prompt, max_new_tokens=5, seed=42
    )

    assert result[:3] == prompt
    assert len(result) == 8
    assert all(0 <= token < 32 for token in result)


def test_reproducible_generation(model):
    first = generate_tokens(
        model, [1, 2], max_new_tokens=8, seed=123
    )

    second = generate_tokens(
        model, [1, 2], max_new_tokens=8, seed=123
    )

    assert first == second


def test_top_one_is_greedy(model):
    first = generate_tokens(
        model, [1], max_new_tokens=5, top_k=1, seed=4
    )

    second = generate_tokens(
        model, [1], max_new_tokens=5, top_k=1, seed=99
    )

    assert first == second


def test_long_prompt_uses_context_window(model):
    prompt = list(range(12))

    result = generate_tokens(
        model, prompt, max_new_tokens=2
    )

    assert result[:12] == prompt
    assert len(result) == 14


@pytest.mark.parametrize(
    "arguments",
    [
        {"temperature": 0},
        {"top_k": 0},
        {"max_new_tokens": -1},
        {"prompt_ids": []},
    ],
)
def test_invalid_generation_arguments(model, arguments):
    options = {"prompt_ids": [1]}
    options.update(arguments)

    with pytest.raises(ValueError):
        generate_tokens(model, **options)


def test_training_mode_restored(model):
    model.train()

    generate_tokens(
        model, [1, 2], max_new_tokens=2
    )

    assert model.training
