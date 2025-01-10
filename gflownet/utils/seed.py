import numpy as np
import random
import torch


def seed_everything(seed: int):
    """
    Seed everything for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)  # Python's random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if used)
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = (
        False  # Disables auto-tuning for better reproducibility
    )
