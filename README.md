# gpt2-gqa-shakespeare

GPT-2-style Transformer implementation with grouped-query attention and BPE tokenization, trained on the Shakespeare dataset.
Explain the decoder-only Transformer architecture, configuration, how Q/K/V attention operates, the use of pre-LayerNorm and residual connections, and how the custom 512-token BPE vocabulary is connected to the embedding and output layers.

Include the commands for reproducing the tests and smoke test. Credit Karpathy's code-along as an educational reference, while documenting your implementation decisions.

# gpt4

GPT-2 Architecture
Explain the decoder-only Transformer architecture, configuration, how Q/K/V attention operates, the use of pre-LayerNorm and residual connections, and how the custom 512-token BPE vocabulary is connected to the embedding and output layers.

Include the commands for reproducing the tests and smoke test. Credit Karpathy's code-along as an educational reference, while documenting your implementation decisions.

# gpt5

Grouped-Query Attention describing the eight query heads, two KV heads, group size of four, attention implementation and parameter comparison.

Include the original GQA research paper as the technical reference.

Document that your implementation uses explicit KV repetition for compatibility. Avoid claiming that it provides native fused GQA execution or an optimized KV cache.

# gpt6

Training Pipeline documenting the data requirements, training commands, parameter settings, validation strategy, checkpoint format, resumption command and CPU limitations.
