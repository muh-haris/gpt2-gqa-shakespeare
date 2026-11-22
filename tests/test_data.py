import pytest

from prepare_data import split_text


def test_split_ratio():
    text = "abcdefghij" * 10

    train, val = split_text(text)

    assert len(train) == 90
    assert len(val) == 10
    assert train + val == text


def test_unicode_preserved():
    text = "Hello 🎭 Shakespeare! " * 10

    train, val = split_text(text)

    assert train + val == text


def test_invalid_split():
    with pytest.raises(ValueError):
        split_text("Hello", train_fraction=1.0)
