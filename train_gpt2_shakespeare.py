"""Test the GPT2 model with pretrained weight from HuggingFace Transformers."""
import torch
import torch.nn.functional as F
from gpt2 import GPT
from dataloader import DataLoaderLite
import tiktoken


def main():
    """If this code runs and produces good coherent text, then the model is working """
    # attempt to auto detect the device including cuda and mps and cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    B, T = 4, 128  # batch size, sequence length
    train_loader = DataLoaderLite(B, T)

    model = GPT.from_pretrained('gpt2')
    model.to(device)

    num_epochs = 20

    optimizer = torch.optim.AdamW(model.parameters())


    for epoch in range(num_epochs):

    
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)
        print(epoch)
        print(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training complete.")

    model.eval()

    import tiktoken

    num_return_sequences = 5
    max_length = 30


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