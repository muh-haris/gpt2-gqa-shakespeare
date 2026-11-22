import pytest

from src.tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    text = "banana bandana banana bandana\n" * 12
    return BPETokenizer.train(text, vocab_size=280)


def test_round_trip(tokenizer):
    sample = "To be, or not to be!\nHello 👋 — Shakespeare.\t"
    ids = tokenizer.encode(sample)

    assert tokenizer.decode(ids) == sample


def test_compression(tokenizer):
    sample = "banana bandana banana"
    ids = tokenizer.encode(sample)

    assert len(ids) < len(sample.encode("utf-8"))


def test_serialization(tmp_path, tokenizer):
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)
    sample = "New words! banana\n"

    assert loaded.encode(sample) == tokenizer.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_consistent_training():
    corpus = "abracadabra abracadabra " * 10

    first = BPETokenizer.train(corpus, vocab_size=275)
    second = BPETokenizer.train(corpus, vocab_size=275)

    assert first.merges == second.merges


def test_invalid_vocab():
    with pytest.raises(ValueError):
        BPETokenizer.train("test", vocab_size=200)


def test_token_id_bounds(tokenizer):
    ids = tokenizer.encode("Shakespeare")

    assert all(0 <= token_id < tokenizer.vocab_size for token_id in ids)
