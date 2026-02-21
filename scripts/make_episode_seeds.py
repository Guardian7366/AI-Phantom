# scripts/make_episode_seeds.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np


def _normalize_path(path: str) -> str:
    p = (path or "").strip().strip('"').strip("'")
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.normpath(p)


def _require_positive_int(name: str, value) -> int:
    try:
        v = int(value)
    except Exception:
        raise ValueError(f"'{name}' debe ser int. Recibido: {value!r}")
    if v <= 0:
        raise ValueError(f"'{name}' debe ser > 0. Recibido: {v}")
    return v


def _make_seeds(
    *,
    episodes: int,
    seed: int,
    lo: int,
    hi: int,
    unique: bool,
    avoid: Optional[List[int]] = None,
) -> List[int]:
    """
    Genera una lista de seeds por episodio de manera reproducible.
    - Usa numpy RNG PCG64 (np.random.default_rng).
    - Rango: [lo, hi] (incluye extremos si hi==lo no permitido).
    - unique=True intenta evitar repeticiones; si el rango es chico puede fallar.
    """
    episodes = int(episodes)
    lo = int(lo)
    hi = int(hi)

    if lo < 0 or hi < 0:
        raise ValueError(f"lo/hi deben ser >= 0. Recibido: lo={lo}, hi={hi}")
    if hi <= lo:
        raise ValueError(f"'hi' debe ser > 'lo'. Recibido: lo={lo}, hi={hi}")

    rng = np.random.default_rng(int(seed))

    avoid_set = set(int(x) for x in (avoid or []))
    out: List[int] = []

    if not unique:
        # Camino más rápido: O(episodes)
        for _ in range(episodes):
            s = int(rng.integers(lo, hi + 1))
            out.append(s)
        return out

    # unique=True (defensivo):
    # - si el rango es suficientemente grande, muestreamos y reintentamos colisiones.
    # - acotamos intentos para evitar loops infinitos.
    range_size = (hi - lo + 1)
    needed = episodes + len(avoid_set)
    if needed > range_size:
        raise ValueError(
            f"unique=True pero rango demasiado pequeño: rango={range_size}, "
            f"necesitas >= {needed} (episodes + avoid)."
        )

    seen = set(avoid_set)
    retries = 0
    max_retries = max(10_000, episodes * 50)

    while len(out) < episodes:
        s = int(rng.integers(lo, hi + 1))
        if s in seen:
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(
                    f"No se pudieron generar seeds únicas con suficientes intentos. "
                    f"Intenta aumentar el rango (hi) o desactivar --unique."
                )
            continue
        seen.add(s)
        out.append(s)

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Genera un JSON list[int] con seeds por episodio para evaluación/sandbox reproducible."
    )
    ap.add_argument("--episodes", type=int, default=300, help="Número de episodios/seeds a generar.")
    ap.add_argument("--seed", type=int, default=123, help="Seed base del generador (reproducibilidad).")

    ap.add_argument("--lo", type=int, default=0, help="Mínimo valor de seed (incl.).")
    ap.add_argument("--hi", type=int, default=9_999_999, help="Máximo valor de seed (incl.).")

    ap.add_argument("--unique", action="store_true", help="Evita seeds repetidas (más estricto).")

    ap.add_argument(
        "--avoid_file",
        type=str,
        default=None,
        help="JSON list[int] de seeds a evitar (ej: las ya usadas en otro set).",
    )

    ap.add_argument(
        "--out",
        type=str,
        default="results/seeds/episode_seeds.json",
        help="Ruta de salida del JSON list[int].",
    )

    ap.add_argument(
        "--meta_out",
        type=str,
        default=None,
        help="(Opcional) Guarda metadata del set (JSON) para auditoría/reproducibilidad.",
    )

    args = ap.parse_args()

    episodes = _require_positive_int("episodes", args.episodes)
    base_seed = int(args.seed)

    lo = int(args.lo)
    hi = int(args.hi)

    avoid = None
    if args.avoid_file is not None:
        af = _normalize_path(args.avoid_file)
        if not os.path.exists(af):
            raise FileNotFoundError(f"No existe avoid_file: {af}")
        with open(af, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("avoid_file debe ser JSON list[int].")
        avoid = [int(x) for x in data]

    seeds = _make_seeds(
        episodes=episodes,
        seed=base_seed,
        lo=lo,
        hi=hi,
        unique=bool(args.unique),
        avoid=avoid,
    )

    out_path = _normalize_path(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(seeds, f, indent=2, ensure_ascii=False)

    print(f"Saved seeds: {out_path} (n={len(seeds)})")

    if args.meta_out:
        meta_path = _normalize_path(args.meta_out)
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        meta = {
            "episodes": int(episodes),
            "seed_base": int(base_seed),
            "lo": int(lo),
            "hi": int(hi),
            "unique": bool(args.unique),
            "avoid_file": None if args.avoid_file is None else _normalize_path(args.avoid_file),
            "out": out_path,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()