"""Implementation of the attention mechanism."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # TODO: Define the number of heads and the embedding dimensionality from the configuration
        self.n_head = config.n_head # number of attention heads
        self.n_embd = config.n_embd # embedding dimension

        # TODO: Define the "c_attn" layer, which is a linear layer that produces the key query and value vectors (hint: it should output 3 times the embedding dimensionality) (one for each of query, key, and value)
        # linear layer to compute query, key, value vectors --> * 3
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # TODO: Define the "c_proj" layer, which is a linear layer that projects the output back to the embedding dimensionality (hint: it should output the same dimensionality as the input) (n_embd)
        # linear layer to project attention output back to embedding size
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        
        # TODO: Calculate the query, key, and value vectors using the c_attn layer then split them into sep
        qkv = self.c_attn(x) # Replace this with your code to calculate qkv
        # computes queries, keys, values in one linear projection
        q, k, v = qkv.split(self.n_embd, dim=2)

        # splits
        # Need to do some tranposing to match the gpt-2 implementation exactly, don't worry about this
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # scaled dot-product attention with causal masking (prevents attending to future tokens).
            # For each token in a sequence, figure out how much it should "pay attention" to other tokens.
            # causal masking --> gpt generates text one token at a time. must Not look at future tokens, only past and current.
        # TODO: Calculate the attention output using scaled dot-product attention from torch.nn.functional (we want causal attention, so set is_causal=True)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal = True) # Replace this with your code to calculate the attention output
        # merge into single embedding per token
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # TODO: Project the output back to the embedding dimensionality using the c_proj layer
        y = self.c_proj(y) # Replace this with your code to project the output back to the embedding dimensionality
        
        return y