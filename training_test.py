from dataloader import DataLoaderLite
import torch
import tiktoken

loader = DataLoaderLite(B=4, T=8)

x, y = loader.next_batch() 
print("x:", x)
print("y:", y)

