# Tiny Shakespeare Dataset

This directory contains the Tiny Shakespeare dataset used to train and evaluate our GPT-2-style language model with Grouped-Query Attention (GQA).

**Dataset source:** [Andrej Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)

The dataset contains approximately 1.1 million characters of Shakespearean dialogue and text.

## Dataset Structure

- `raw/shakespeare.txt` — Original downloaded dataset.
- `processed/train.txt` — Training data (90%).
- `processed/val.txt` — Validation data (10%).

## Data Preparation

Run the following command from the project root:

`python prepare_data.py`

This downloads the dataset and creates the training and validation files.

## Git Tracking

The dataset files are excluded from Git because they can be downloaded and regenerated.

Only this `README.md` should be committed to document the dataset and its preparation process.
