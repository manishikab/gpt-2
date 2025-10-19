from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from block import Block

# ------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024  # context size
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # TODO: Define the transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # wte. word/token embeddings: look up table-- holds all vectors that correspond to token values
            wpe = nn.Embedding(config.block_size, config.n_embd), # wpe. word positions embeddings: tell model where word is
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # tranformer blocks (attention and feed forward)
            ln_f = nn.LayerNorm(config.n_embd) # layer normalization
        ))


        # TODO: Define the linear net (matrix)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # converts input (hidden states / vectors) into predictions over the vocabulary


        # Weight sharing scheme: tie the input and output embeddings ##ADDDED LATER##
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize the weights
        self.apply(self._init_weights)

    # ADDED LATER
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Layernorm also need to be initialized, but we like the default PyTorch initialization

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward sequence of length %d, block size is only %d" % (T, self.config.block_size)
        # forward the GPT model

        # TODO: Add the forward logic
        # Positions IDs
        pos = torch.arange(T, device=idx.device, dtype=torch.long) #1d vector of position indices
        # Learned positional embeddings
        pos_emb = self.transformer.wpe(pos) # look up position embeddings for each index
        # Token embeddings
        tok_emb = self.transformer.wte(idx) # look up token embeddings for each token id
        # Combine embeddings
        x = tok_emb + pos_emb # each token rep includes what the token is and where it is
        # Forward through the transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # TODO: Final layer normalization

        x = self.transformer.ln_f(x) # for each token, rescale to make it eaier to interpret later
        # output shape is (B, T, n_embd) == (Batch, Sequence Length, Hidden Size)



        # Output logits for the vocabulary
        logits = self.lm_head(x)
        ### ADDED LOSS PORTION ###
        
        if targets is not None:
            # Need to flatten out the logits and targets for cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                print(k, sd_hf[k].shape[::-1], sd[k].shape)
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
