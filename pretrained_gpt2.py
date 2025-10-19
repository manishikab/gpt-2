"""Test the GPT2 model with pretrained weight from HuggingFace Transformers."""
import torch
import torch.nn.functional as F
from gpt2 import GPT

def main():
    """If this code runs and produces good coherent text, then the model is working and we have nearly identically the same model as the HuggingFace Transformers GPT2 model."""
    # attempt to auto detect the device including cuda and mps and cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    model = GPT.from_pretrained('gpt2')
    print("Model loaded successfully.")

    num_return_sequences = 5
    max_length = 30

    model.eval()
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a large language model.")
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    tokens = tokens.repeat(num_return_sequences, 1)  # Repeat for num_return_sequences
    x = tokens.to(device)  # Move to the same device as the model

    torch.manual_seed(42)  # For reproducibility
    if device.type == 'mps':
        # MPS backend requires manual seed setting
        torch.mps.manual_seed(42)  # For reproducibility on MPS
    elif device.type == 'cuda':
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            xcol = torch.gather(topk_indices, dim=-1, index=ix)
            x = torch.cat((x, xcol), dim=1)

    # Decode the generated tokens
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

if __name__ == "__main__":
    main()
    print("Nothing crashed, everything is fine.")