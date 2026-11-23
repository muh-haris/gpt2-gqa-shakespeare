# Final Evaluation — GPT-2 with Grouped-Query Attention

## 1. Project Objective

The objective was to implement a compact GPT-2-style decoder-only Transformer with grouped-query attention (GQA), develop a byte-level BPE tokenizer and train the resulting language model on the Tiny Shakespeare dataset.

The implementation uses PyTorch and was developed and tested on Windows 10.

## 2. Model Architecture

The model contains four Transformer blocks with an embedding dimension of 256 and a maximum context length of 128 tokens.

Grouped-query attention uses eight query heads and two key/value heads. Each KV head serves four query heads.

The resulting model contains 2,928,640 parameters.

## 3. Dataset and Tokenization

The Tiny Shakespeare dataset contains 1,115,394 characters and was divided into 90% training and 10% validation data.

A byte-level BPE tokenizer was trained exclusively on the training partition. The final vocabulary contains 512 tokens, including 256 base byte tokens and 256 learned merges.

Tokenization reduced the training corpus to 621,873 tokens and the validation corpus to 70,184 tokens.

## 4. Multi-Head Attention vs Grouped-Query Attention

The standard multi-head attention model contains 3,323,392 parameters.

The equivalent grouped-query attention model contains 2,928,640 parameters, representing a reduction of 394,752 parameters, or approximately 11.88%.

This demonstrates a reduction in parameter count, not a measured improvement in inference speed or predictive quality.

## 5. Training Results

The GQA model was trained for 500 optimizer steps using AdamW with learning-rate warmup and cosine decay.

The initial sampled validation loss was 6.3038.

The lowest recorded sampled validation loss was 3.3982 at step 500.

Validation loss decreased across the recorded evaluation checkpoints.

## 6. Text Generation

Text generation was implemented using autoregressive sampling with configurable temperature and top-k filtering.

Samples were generated using a `ROMEO:` prompt at temperatures 0.6, 0.9 and 1.2.

The outputs contained recognizable words and some dialogue-like structure. However, they also exhibited repetition, malformed words and limited grammatical coherence.

The model therefore demonstrated functional text generation without achieving fluent Shakespeare-style writing.

## 7. Testing

The complete automated test suite contained 41 tests, all of which passed.

The tests covered dataset preparation, BPE tokenization, model architecture, causal masking, grouped-query attention, training utilities, checkpointing and text generation.

## 8. Limitations

The model was trained from scratch on a relatively small corpus. Its 128-token context window and limited training duration restrict the quality of generated passages.

The evaluation uses sampled validation windows rather than exhaustive validation-set scoring. No separately trained MHA quality baseline or inference-speed benchmark was performed.

The GQA implementation explicitly expands KV heads during attention computation and does not implement an optimized KV cache.

## 9. Conclusion

The project demonstrates an end-to-end implementation of a GPT-2-style language model with a custom BPE tokenizer and grouped-query attention.

The primary demonstrated achievements are a functioning training and inference pipeline, successful automated verification, decreasing validation loss and reduced model parameter count through GQA.

Longer training, stronger evaluation and optimized inference remain potential future improvements.
