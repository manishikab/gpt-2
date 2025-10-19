"""Attention Block for Transformer model."""

import torch
import torch.nn as nn
from attention import CausalSelfAttention
from mlp import MLP

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # TODO: Define the two layer normalization layers and the attention layer
        self.ln_1 = nn.LayerNorm(config.n_embd) # Layer normalization before attention
        self.attn = CausalSelfAttention(config) # Causal self-attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd) # Second layer normalization before MLP
        
        # Need to define custom MLP because it is not just one feedforward layer but many in paralell for each token.
        # TODO: Define the MLP layer (Multi Layer Perceptron - fully connected feedforward NN)
        self.mlp = MLP(config) # Feedforward MLP

    def forward(self, x):
        # TODO: Feedforward the result through the attention layer and the MLP layer to get the final output
        x = x + self.attn(self.ln_1(x)) # Residual connection around attention
        x = x + self.mlp(self.ln_2(x)) # Residual connection around MLP
        return x