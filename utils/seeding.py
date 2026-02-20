import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinismo (puede bajar rendimiento un poco)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
