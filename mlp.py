"""MLP Module for GPT-2 Transformer Block"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Define the fully connected layer for the MLP (hint: it should output 4 times the embedding dimensionality)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # TODO: Define the projection layer for the MLP (hint: it should output the same dimensionality as the input) (n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # Define a GeLU activation function. We useonly considering previous and current GeLU as it is the activation function used in GPT-2
        self.gelu = nn.GELU()

    def forward(self, x):
        # TODO: Feedforward the input through the fully connected layer, apply the GeLU activation, then project back to the embedding dimensionality
        # 1. Pass input through first linear layer (expand)
        x = self.c_fc(x)
        # 2. Apply GeLU nonlinearity
        x = self.gelu(x)
        # 3. Pass through projection layer (compress back)
        x = self.c_proj(x)
        return x