# ai_phantom/core/compile_utils.py
from __future__ import annotations

from typing import Optional

import os
import torch


def _triton_available() -> bool:
    """
    torch.compile (inductor) suele requerir triton para kernels en GPU.
    En Windows muchas veces no está o no funciona bien.
    """
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def safe_torch_compile(
    model: torch.nn.Module,
    device: torch.device,
    example_input: Optional[torch.Tensor] = None,
    enable_env_var: str = "AI_PHANTOM_COMPILE",
) -> torch.nn.Module:
    """
    Compila el modelo con torch.compile SOLO si:
      - device es CUDA
      - AI_PHANTOM_COMPILE=1
      - Triton está disponible
      - y el warmup no falla

    Si algo falla, retorna el modelo original (eager) sin romper el entrenamiento.
    """
    want = os.getenv(enable_env_var, "0").strip() == "1"
    if not want:
        return model

    if device.type != "cuda":
        print("ℹ️ torch.compile skipped (device != cuda)")
        return model

    if not _triton_available():
        print("⚠️ torch.compile skipped: Triton no disponible/funcional en este entorno.")
        return model

    # Import aquí para no tocar nada si no se usa
    import torch._dynamo  # type: ignore

    # Si algo falla durante la compilación/ejecución, que no mate el proceso
    torch._dynamo.config.suppress_errors = True

    try:
        compiled = torch.compile(model)  # lazy compile: se materializa en el 1er forward

        # Warmup para forzar la compilación AHORA (y no tronar en medio del loop)
        if example_input is not None:
            compiled.eval()
            with torch.no_grad():
                _ = compiled(example_input)

        print("✅ torch.compile enabled (safe)")
        return compiled
    except Exception as e:
        print(f"⚠️ torch.compile failed -> fallback eager. Reason: {e}")
        return model