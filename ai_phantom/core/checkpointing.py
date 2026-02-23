from __future__ import annotations

from typing import Any, Dict, Optional
import time
import random

import numpy as np
import torch


def _get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda_all"] = None
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    try:
        if "python_random" in state and state["python_random"] is not None:
            random.setstate(state["python_random"])
        if "numpy_random" in state and state["numpy_random"] is not None:
            np.random.set_state(state["numpy_random"])
        if "torch_cpu" in state and state["torch_cpu"] is not None:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda_all", None) is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        # no tronamos si el estado no es compatible entre máquinas/torch versions
        print(f"⚠️ RNG restore skipped (incompatible state): {e}")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra: Optional[Dict[str, Any]] = None,
    *,
    save_rng: bool = True,
) -> None:
    payload: Dict[str, Any] = {"model_state": model.state_dict()}

    if optimizer is not None:
        payload["optim_state"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = extra

    # ✅ D1: guardar RNG + meta para reproducibilidad
    if save_rng:
        payload["rng_state"] = _get_rng_state()
    payload["meta"] = {
        "time_unix": float(time.time()),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }

    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device | None = None,
    *,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optim_state" in payload:
        optimizer.load_state_dict(payload["optim_state"])

    # ✅ D1: opcional restaurar RNG
    if restore_rng and ("rng_state" in payload):
        _set_rng_state(payload["rng_state"])

    return payload.get("extra", {})