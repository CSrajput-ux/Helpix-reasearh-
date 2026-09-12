import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds across standard library random, NumPy, and PyTorch.
    Configures cuDNN deterministic flags. Note that full bit-for-bit determinism
    on GPU may depend on CUDA library version and atomic operation nondeterminism.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
