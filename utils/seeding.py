import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Solo fija seeds. NO toca flags de cudnn/TF32/determinismo.
    Eso lo controla scripts/train.py en _apply_torch_determinism() (Ley 1 y Ley 4).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)