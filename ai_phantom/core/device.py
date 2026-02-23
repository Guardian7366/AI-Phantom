from __future__ import annotations

from dataclasses import dataclass
import os
import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    allow_tf32: bool = True
    cudnn_benchmark: bool = True


def select_device(device: str = "auto", allow_tf32: bool = True, cudnn_benchmark: bool = True) -> DeviceConfig:
    """
    Selección de device consistente para todo el proyecto.
    - device: "auto" | "cpu" | "cuda"
    """
    if device == "auto":
        use_cuda = torch.cuda.is_available()
        dev = torch.device("cuda" if use_cuda else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se pidió device='cuda' pero CUDA no está disponible.")
        dev = torch.device("cuda")
    elif device == "cpu":
        dev = torch.device("cpu")
    else:
        raise ValueError(f"device inválido: {device}")

    # Flags de rendimiento (seguros por defecto)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    # Opcional: limitar threads CPU si quieres reproducibilidad/estabilidad
    # os.environ.setdefault("OMP_NUM_THREADS", "1")

    return DeviceConfig(device=dev, allow_tf32=allow_tf32, cudnn_benchmark=cudnn_benchmark)