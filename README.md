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

# reproducibility commands

These commands in the README so another developer can run the project after cloning.
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Document the separate PyTorch installation command if requirements.txt does not include it. The README must make clear that PyTorch is required.

Then:
python prepare_data.py
python train_tokenizer.py
python -m pytest -v

# To reproduce your 500-step experiment:

python train.py --steps 500 --batch-size 4 --seq-len 64 --warmup-steps 25 --eval-every 50 --eval-batches 10 --log-every 10 --save-every 50 --threads 4 --seed 42

# To generate text after training:

python generate.py --prompt "ROMEO:" --max-new-tokens 120 --temperature 0.9 --top-k 40 --seed 42

# Include a README section explaining where to place the downloaded files:

checkpoints/gpt_gqa_best.pt
data/processed/tokenizer.json
