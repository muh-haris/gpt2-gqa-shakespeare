# Tiny Shakespeare Dataset

Source: Andrej Karpathy's char-rnn repository.

Run the preparation script from the project root:

python prepare_data.py

This creates:

- data/raw/shakespeare.txt
- data/processed/train.txt
- data/processed/val.txt

The corpus is divided chronologically into
90% training text and 10% validation text.

The BPE tokenizer will be trained only on
the training portion to avoid data leakage.
