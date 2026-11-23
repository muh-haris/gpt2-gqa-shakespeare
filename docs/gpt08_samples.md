# GPT-08: Shakespeare Text Generation

## Model

- Architecture: GPT-2-style Transformer with GQA
- Parameters: 2,928,640
- Vocabulary size: 512
- Training steps: 500
- Best validation loss: 3.3982
- Checkpoint: gpt_gqa_best.pt
- Device: CPU

## Generation settings

- Prompt: ROMEO:
- New tokens: 120
- Top-k: 40
- Random seed: 42

## Sample A — Temperature 0.6

Paste the actual generated text here.

## Sample B — Temperature 0.9

Paste the actual generated text here.

## Sample C — Temperature 1.2

Paste the actual generated text here.

## Observations

Describe:

- Readability
- Dialogue formatting
- Repetition
- Grammatical coherence
- Differences between sampling temperatures

## Limitations

The model was trained from scratch using a small
Shakespeare corpus and 500 optimizer updates.

Generation is limited to a 128-token context window.

Generated text is assessed qualitatively rather than
through an automatic language-quality benchmark.
