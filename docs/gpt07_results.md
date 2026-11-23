# GPT-07: Shakespeare Model Training Results

## Experiment setup

- Dataset: Tiny Shakespeare
- Tokenizer: Custom byte-level BPE
- Vocabulary size: 512
- Model: GPT-2-style decoder-only Transformer with GQA
- Model parameters: 2,928,640
- Query heads: 8
- Key/value heads: 2
- Optimizer: AdamW
- Training device: CPU
- Random seed: 42

## Training configuration

- Total steps: [actual value]
- Batch size: [actual value]
- Sequence length: [actual value]
- Maximum learning rate: [actual value]
- Validation interval: [actual value]

## Results

- Initial validation loss: [value]
- Final validation loss: [value]
- Best validation loss: [value]
- Best checkpoint step: [step]

## Observations

[Describe the validation-loss trend, whether improvement slowed and any evidence of overfitting.]

## Limitations

The experiment uses a small Shakespeare corpus and a compact Transformer trained from scratch. Validation loss is estimated using sampled windows rather than an exhaustive pass over the validation corpus. Further training and comparison against a standard multi-head attention baseline are required before making claims about model quality or computational advantages.

## Reproducibility

[Record Python version, PyTorch version, training command and Git commit.]

## Next step

Generate Shakespeare-style text from the best checkpoint and examine the output qualitatively.
