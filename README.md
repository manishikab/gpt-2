**GPT-2 From Scratch (PyTorch)**
A complete, from-scratch implementation of a GPT-2–style Transformer language model built in PyTorch.
This project reconstructs the GPT-2 architecture end-to-end, including tokenization, self-attention, training, and text generation, with minimal reliance on high-level abstractions.

Project Overview:

- Implemented a GPT-2–style Transformer from first principles

- Reproduced core architectural components: self-attention, MLPs, residual connections, and layer normalization

Trained the model on a real text corpus (Shakespeare, Wikipedia) and generated coherent text samples

- Validated correctness using pretrained GPT-2 weights

Model Structure:

- gpt2.py – Top-level GPT-2 model definition

- block.py – Transformer block (Attention + MLP)

- attention.py – Multi-head causal self-attention

- mlp.py – Feed-forward network
