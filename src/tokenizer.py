import json
import re
from collections import Counter
from pathlib import Path


def merge_pair(tokens, pair, new_id):
    result = []
    i = 0

    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return tuple(result)


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.vocab = [bytes([i]) for i in range(256)]
        self.ranks = {}

        for index, pair in enumerate(self.merges):
            if len(pair) != 2 or any(
                not isinstance(x, int) or x < 0 or x >= 256 + index
                for x in pair
            ):
                raise ValueError("Invalid BPE merge rule")

            self.ranks[pair] = index
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size=512):
        if vocab_size < 256:
            raise ValueError("Vocabulary must include all 256 bytes")
        if not text:
            raise ValueError("Training text is empty")

        # Count repeated chunks once, keeping their frequencies.
        chunks = Counter(
            tuple(part)
            for part in re.findall(rb"\S+|\s+", text.encode("utf-8"))
        )

        merges = []

        for _ in range(vocab_size - 256):
            pair_counts = Counter()

            for tokens, frequency in chunks.items():
                for pair in zip(tokens, tokens[1:]):
                    pair_counts[pair] += frequency

            if not pair_counts:
                break

            # Prefer the most frequent pair. Resolve ties consistently.
            best = max(
                pair_counts,
                key=lambda pair: (
                    pair_counts[pair], -pair[0], -pair[1]
                )
            )

            new_id = 256 + len(merges)
            merges.append(best)

            updated = Counter()

            for tokens, frequency in chunks.items():
                updated[merge_pair(tokens, best, new_id)] += frequency

            chunks = updated

            if len(merges) % 64 == 0:
                print(f"Learned {len(merges)} merges")

        return cls(merges)

    def encode(self, text):
        ids = []

        for part in re.findall(rb"\S+|\s+", text.encode("utf-8")):
            tokens = tuple(part)

            while len(tokens) > 1:
                choices = [
                    (self.ranks[pair], pair)
                    for pair in zip(tokens, tokens[1:])
                    if pair in self.ranks
                ]

                if not choices:
                    break

                rank, pair = min(choices)
                tokens = merge_pair(tokens, pair, 256 + rank)

            ids.extend(tokens)

        return ids

    def decode(self, ids, errors="strict"):
        for token_id in ids:
            if not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Invalid token ID: {token_id}")

        raw = b"".join(self.vocab[token_id] for token_id in ids)
        return raw.decode("utf-8", errors=errors)

    def save(self, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "merges": [list(pair) for pair in self.merges]
        }

        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, filename):
        payload = json.loads(
            Path(filename).read_text(encoding="utf-8")
        )

        if payload.get("version") != 1:
            raise ValueError("Unsupported tokenizer file")

        return cls(payload["merges"])
