import torch

def get_device_and_seed():
    """Detects the available device (MPS, CUDA, or CPU) and sets the random seed."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if device.type == 'mps':
        torch.mps.manual_seed(seed)
    elif device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    return device