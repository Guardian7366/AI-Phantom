# environments/maze/maze_families.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, Tuple
import numpy as np

Coord = Tuple[int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def gen_bernoulli(
    rng: np.random.Generator,
    *,
    size: int,
    wall_prob: float,
    edge_wall_scale: float = 0.6,
) -> np.ndarray:
    """
    Familia 1: Bernoulli clásico (lo que ya usas).
    """
    p = float(wall_prob)
    p_edge = p * float(edge_wall_scale)

    grid = (rng.random((size, size)) < p).astype(np.int8)

    # bordes un poco menos densos (como tu versión)
    grid[0, :]  = (rng.random(size) < p_edge).astype(np.int8)
    grid[-1, :] = (rng.random(size) < p_edge).astype(np.int8)
    grid[:, 0]  = (rng.random(size) < p_edge).astype(np.int8)
    grid[:, -1] = (rng.random(size) < p_edge).astype(np.int8)

    # seguridad: esquinas libres
    grid[0, 0] = 0
    grid[size - 1, size - 1] = 0
    return grid


def gen_rooms(
    rng: np.random.Generator,
    *,
    size: int,
    wall_prob: float,
    rooms_min: int = 2,
    rooms_max: int = 5,
    room_min_size: int = 2,
    room_max_size: int = 4,
    corridor_width: int = 1,
) -> np.ndarray:
    """
    Familia 2: "Rooms & corridors" muy barata.
    Idea:
      - Empieza con todo muro
      - Cava varios cuartos (rectángulos libres)
      - Conecta centros con corredores Manhattan
    Nota: no busca ser perfecto; busca diversidad estructural.
    """
    # base: todo muro
    grid = np.ones((size, size), dtype=np.int8)

    n_rooms = int(rng.integers(int(rooms_min), int(rooms_max) + 1))
    centers = []

    for _ in range(n_rooms):
        h = int(rng.integers(int(room_min_size), int(room_max_size) + 1))
        w = int(rng.integers(int(room_min_size), int(room_max_size) + 1))

        # colocar dentro de bordes
        r0 = int(rng.integers(1, max(2, size - h - 1)))
        c0 = int(rng.integers(1, max(2, size - w - 1)))

        grid[r0:r0 + h, c0:c0 + w] = 0
        centers.append((r0 + h // 2, c0 + w // 2))

    # conecta centros en cadena
    def carve_corridor(a: Coord, b: Coord):
        ar, ac = a
        br, bc = b
        # orden aleatorio: horizontal-first o vertical-first
        if bool(rng.integers(0, 2)):
            # horizontal
            step = 1 if bc >= ac else -1
            for c in range(ac, bc + step, step):
                for dw in range(-(corridor_width // 2), corridor_width // 2 + 1):
                    cc = c + dw
                    if 0 <= ar < size and 0 <= cc < size:
                        grid[ar, cc] = 0
            # vertical
            step = 1 if br >= ar else -1
            for r in range(ar, br + step, step):
                for dw in range(-(corridor_width // 2), corridor_width // 2 + 1):
                    rr = r + dw
                    if 0 <= rr < size and 0 <= ac < size:
                        grid[rr, ac] = 0
        else:
            # vertical
            step = 1 if br >= ar else -1
            for r in range(ar, br + step, step):
                for dw in range(-(corridor_width // 2), corridor_width // 2 + 1):
                    rr = r + dw
                    if 0 <= rr < size and 0 <= ac < size:
                        grid[rr, ac] = 0
            # horizontal
            step = 1 if bc >= ac else -1
            for c in range(ac, bc + step, step):
                for dw in range(-(corridor_width // 2), corridor_width // 2 + 1):
                    cc = c + dw
                    if 0 <= ar < size and 0 <= cc < size:
                        grid[ar, cc] = 0

    for i in range(1, len(centers)):
        carve_corridor(centers[i - 1], centers[i])

    # agrega algo de ruido controlado: abre celdas con wall_prob bajo
    # (esto evita “cuartos perfectos” siempre)
    p_open = float(wall_prob) * 0.35
    mask = (rng.random((size, size)) < p_open)
    grid[mask] = 0

    # seguridad
    grid[0, 0] = 0
    grid[size - 1, size - 1] = 0
    return grid


def gen_corridors(
    rng: np.random.Generator,
    *,
    size: int,
    wall_prob: float,
    walkers: int = 3,
    walk_steps_scale: float = 2.2,
) -> np.ndarray:
    """
    Familia 3: "Corridors" por random-walk carving.
    - Empieza con muros
    - Varios walkers que van cavando caminos
    """
    grid = np.ones((size, size), dtype=np.int8)

    steps = int(max(1, int(walk_steps_scale * size * size)))
    walkers = int(max(1, walkers))

    # inicializa walkers en posiciones aleatorias
    pos = []
    for _ in range(walkers):
        r = int(rng.integers(0, size))
        c = int(rng.integers(0, size))
        pos.append([r, c])
        grid[r, c] = 0

    for _ in range(steps):
        wi = int(rng.integers(0, walkers))
        r, c = pos[wi]

        # paso aleatorio
        dr, dc = [(1,0), (-1,0), (0,1), (0,-1)][int(rng.integers(0, 4))]
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            pos[wi][0], pos[wi][1] = nr, nc
            grid[nr, nc] = 0

    # densidad: si quedó demasiado abierto/cerrado, ajusta con wall_prob suave
    # (abre algunos muros al azar para evitar desconexiones totales)
    p_open = float(wall_prob) * 0.25
    mask = (rng.random((size, size)) < p_open)
    grid[mask] = 0

    grid[0, 0] = 0
    grid[size - 1, size - 1] = 0
    return grid


# Registro oficial
FAMILY_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "bernoulli": gen_bernoulli,
    "rooms": gen_rooms,
    "corridors": gen_corridors,
}