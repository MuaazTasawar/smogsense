import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds across numpy, random, and (if installed) torch.

    Ensures reproducible train/val/test splits and model fits across runs.

    Args:
        seed: The seed value to apply everywhere. Defaults to module SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass