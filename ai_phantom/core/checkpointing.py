from __future__ import annotations

from typing import Any, Dict, Optional
import time
import random
import os
import numpy as np
import torch


def _to_cpu_rng_tensor(x: Any) -> Any:
    """
    Asegura tensores RNG en CPU y dtype=uint8 (ByteTensor) para compatibilidad con torch.set_rng_state.
    """
    if torch.is_tensor(x):
        return x.detach().to(device="cpu", dtype=torch.uint8).contiguous().clone()
    return x


def _ensure_uint8_cpu_tensor(x: Any) -> Optional[torch.Tensor]:
    """
    Convierte estados RNG a torch.ByteTensor en CPU.
    Soporta:
      - torch.Tensor (cualquier dtype/device)
      - bytes/bytearray
      - list/tuple de ints (0..255)
      - numpy arrays
    """
    if x is None:
        return None

    try:
        if torch.is_tensor(x):
            return x.detach().to(device="cpu", dtype=torch.uint8).contiguous()

        if isinstance(x, (bytes, bytearray)):
            return torch.tensor(list(x), dtype=torch.uint8, device="cpu").contiguous()

        if isinstance(x, (list, tuple)):
            # lista de ints/uint8
            return torch.tensor(x, dtype=torch.uint8, device="cpu").contiguous()

        if isinstance(x, np.ndarray):
            arr = x.astype(np.uint8, copy=False)
            # from_numpy comparte memoria; contiguous() asegura layout estable
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
            state["torch_cuda_all"] = [_to_cpu_rng_tensor(s) for s in cuda_states]
        except Exception as e:
            print(f"⚠️ CUDA RNG capture skipped: {e}")
            state["torch_cuda_all"] = None
    else:
        state["torch_cuda_all"] = None

    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    """
    Restauración RNG robusta:
    - CPU torch: siempre intenta si es convertible a ByteTensor CPU.
    - CUDA: SOLO restaura si viene como list/tuple de estados y len == device_count().
      Cualquier otro formato se ignora (para evitar restores inválidos).
    """
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
            # ✅ Solo aceptamos lista/tupla de estados CUDA
            if isinstance(cuda_all, (list, tuple)):
                fixed_list = [_ensure_uint8_cpu_tensor(s) for s in cuda_all]  # type: ignore[arg-type]
                fixed = [t for t in fixed_list if t is not None]

                ndev = int(torch.cuda.device_count())
                if len(fixed) == ndev:
                    torch.cuda.set_rng_state_all(fixed)
                else:
                    print(f"⚠️ CUDA RNG restore skipped (len={len(fixed)} != device_count={ndev}).")
            else:
                # tensor único/bytes/ndarray/etc => formato no confiable para set_rng_state_all
                print("⚠️ CUDA RNG restore skipped (unexpected format; expected list/tuple of per-device states).")

    except Exception as e:
        print(f"⚠️ RNG restore skipped (incompatible state): {e}")


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """
    Mueve el estado del optimizador al device del modelo.
    Útil cuando se carga un checkpoint guardado en CPU y luego se entrena en CUDA.
    """
    def move(x: Any) -> Any:
        if torch.is_tensor(x):
            # non_blocking ayuda si el tensor fuente está en pinned mem / staging
            return x.to(device=device, non_blocking=True)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            y = [move(v) for v in x]
            return type(x)(y)  # conserva list/tuple
        return x

    for st in optimizer.state.values():
        for k, v in list(st.items()):
            st[k] = move(v)


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
        # ✅ Guardar optim_state en CPU SIN mutar el optimizador vivo
        optim_state = optimizer.state_dict()
        try:
            for st in optim_state.get("state", {}).values():
                for k, v in list(st.items()):
                    if torch.is_tensor(v):
                        st[k] = v.detach().to(device="cpu").contiguous()
        except Exception as e:
            print(f"⚠️ Optimizer state CPU copy skipped: {e}")

        payload["optim_state"] = optim_state

    if extra is not None:
        payload["extra"] = extra

    # ✅ reproducibilidad
    if save_rng:
        payload["rng_state"] = _get_rng_state()

    payload["meta"] = {
        "time_unix": float(time.time()),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

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
        try:
            model_device = next(model.parameters()).device
            _optimizer_to_device(optimizer, model_device)
        except Exception as e:
            print(f"⚠️ Optimizer state device move skipped after load: {e}")

    if restore_rng and ("rng_state" in payload):
        _set_rng_state(payload["rng_state"])

    return payload.get("extra", {})