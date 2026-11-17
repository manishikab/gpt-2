# GPT-2 Architecture in PyTorch

An educational reconstruction of the GPT-2 transformer architecture in PyTorch. The project implements token and positional embeddings, causal multi-head self-attention, residual transformer blocks, MLPs, layer normalization, tied input/output embeddings, autoregressive generation, and a lightweight training loop.

The implementation can import the published GPT-2 checkpoints from Hugging Face into the custom architecture. That weight-mapping step is the primary correctness check: compatible state dictionaries produce coherent generations using this project's PyTorch model.

## What is implemented
- GPT-2 small model configuration and forward pass
- Causal multi-head self-attention via PyTorch scaled dot-product attention
- Pre-norm residual transformer blocks with GELU MLPs
- Weight tying between token embeddings and the language-model head
- Import of GPT-2 / GPT-2-medium / GPT-2-large / GPT-2-xl weights from Hugging Face
- Autoregressive top-*k* text sampling
- A simple sequential-token data loader and fine-tuning loop
- Streamlit text-generation demo

## Quick start
Create a virtual environment, install the dependencies, and validate the custom model against imported GPT-2 weights:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pretrained_gpt2.py
```

The first run downloads the public Hugging Face GPT-2 checkpoint.

To launch the demo:
```bash
streamlit run app.py
```

To run the fine-tuning experiment:
```bash
python train_gpt2.py
```

`train_gpt2.py` reads `wikitext2_train.txt`, which is included for the small local experiment. Its hyperparameters are modest and meant for learning rather than benchmarking.

## Repository guide

| File | Purpose |
| --- | --- |
| `gpt2.py` | Model configuration, GPT-2 forward pass, initialization, and pretrained-weight import |
| `attention.py` | Causal multi-head self-attention |
| `block.py` | Transformer block with residual connections |
| `mlp.py` | Feed-forward network within each block |
| `dataloader.py` | Sequential next-token batch loader |
| `pretrained_gpt2.py` | Generates text with imported GPT-2 weights |
| `train_gpt2.py` | Fine-tuning experiment on local text data |
| `app.py` | Streamlit text-generation interface |
