from __future__ import annotations

from typing import Any, Dict, Optional
import time
import random
import os
import numpy as np
import torch


def _to_cpu_rng_tensor(x: Any) -> Any:
    # Asegura tensores RNG en CPU y dtype=uint8 (ByteTensor) para compatibilidad con torch.set_rng_state
    if torch.is_tensor(x):
        return x.detach().to(device="cpu", dtype=torch.uint8).clone()
    return x

def _ensure_uint8_cpu_tensor(x: Any) -> Optional[torch.Tensor]:
    """
    Convierte estados RNG a torch.ByteTensor en CPU.
    Soporta:
      - torch.Tensor (cualquier dtype/device)
      - bytes/bytearray
      - list/tuple de ints
      - numpy arrays
    """
    if x is None:
        return None

    try:
        if torch.is_tensor(x):
            return x.detach().to(device="cpu", dtype=torch.uint8).contiguous()

        if isinstance(x, (bytes, bytearray)):
            # torch.tensor(list(bytes)) es seguro y estable
            return torch.tensor(list(x), dtype=torch.uint8, device="cpu").contiguous()

        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=torch.uint8, device="cpu").contiguous()

        if isinstance(x, np.ndarray):
            # Fuerza uint8 y copia segura a CPU tensor
            arr = x.astype(np.uint8, copy=False)
            return torch.from_numpy(arr).to(device="cpu", dtype=torch.uint8).contiguous()

    except Exception:
        return None

    return None

def _get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": _to_cpu_rng_tensor(torch.get_rng_state()),
    }

    if torch.cuda.is_available():
        try:
            cuda_states = torch.cuda.get_rng_state_all()
            # guardar lista de tensores en CPU
            state["torch_cuda_all"] = [_to_cpu_rng_tensor(s) for s in cuda_states]
        except Exception as e:
            print(f"⚠️ CUDA RNG capture skipped: {e}")
            state["torch_cuda_all"] = None
    else:
        state["torch_cuda_all"] = None

    return state

def _set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return

    try:
        py = state.get("python_random", None)
        if py is not None:
            random.setstate(py)

        nr = state.get("numpy_random", None)
        if nr is not None:
            np.random.set_state(nr)

        tc = state.get("torch_cpu", None)
        tc_t = _ensure_uint8_cpu_tensor(tc)
        if tc_t is not None:
            torch.set_rng_state(tc_t)

        cuda_all = state.get("torch_cuda_all", None)
        if torch.cuda.is_available() and (cuda_all is not None):
            # compat: puede venir como tensor único o lista/tupla
            if torch.is_tensor(cuda_all) or isinstance(cuda_all, (bytes, bytearray, list, tuple, np.ndarray)):
                if torch.is_tensor(cuda_all):
                    fixed_list = [_ensure_uint8_cpu_tensor(cuda_all)]
                elif isinstance(cuda_all, (bytes, bytearray, np.ndarray)):
                    fixed_list = [_ensure_uint8_cpu_tensor(cuda_all)]
                else:
                    fixed_list = [_ensure_uint8_cpu_tensor(s) for s in cuda_all]  # type: ignore[arg-type]

                fixed = [t for t in fixed_list if t is not None]
                if len(fixed) > 0:
                    torch.cuda.set_rng_state_all(fixed)

    except Exception as e:
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
        # ✅ Asegura que el estado del optimizador quede en el mismo device que el modelo
        try:
            model_device = next(model.parameters()).device
            for st in optimizer.state.values():
                for k, v in list(st.items()):
                    if torch.is_tensor(v):
                        st[k] = v.to(device=model_device)
        except Exception as e:
            print(f"⚠️ Optimizer state device move skipped: {e}")

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

    # ✅ asegurar que el directorio existe
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ✅ guardado atómico (anti corrupción)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device | None = None,
    *,
    restore_rng: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(f"Checkpoint inválido o corrupto: {path}")

    model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optim_state" in payload:
        optimizer.load_state_dict(payload["optim_state"])

    # ✅ D1: opcional restaurar RNG
    if restore_rng and ("rng_state" in payload):
        _set_rng_state(payload["rng_state"])

    return payload.get("extra", {})