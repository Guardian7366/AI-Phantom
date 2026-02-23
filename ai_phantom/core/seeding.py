from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Semillas globales para reproducibilidad.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Opcional (más determinismo, menos velocidad):
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"