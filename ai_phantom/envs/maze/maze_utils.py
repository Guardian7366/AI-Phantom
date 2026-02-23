from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

Pos = Tuple[int, int]


def in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def neighbors_4(r: int, c: int) -> List[Pos]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def bfs_distance_map(walls: np.ndarray, goal: Pos) -> np.ndarray:
    """
    Devuelve un mapa de distancias BFS desde cada celda libre hacia goal.
    walls: bool array (H,W) True=pared
    """
    h, w = walls.shape
    dist = np.full((h, w), fill_value=-1, dtype=np.int32)

    gr, gc = goal
    if walls[gr, gc]:
        return dist

    q = deque([(gr, gc)])
    dist[gr, gc] = 0

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors_4(r, c):
            if not in_bounds(nr, nc, h, w):
                continue
            if walls[nr, nc]:
                continue
            if dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def sample_free_cell(rng: np.random.Generator, walls: np.ndarray) -> Pos:
    h, w = walls.shape
    free = np.argwhere(~walls)
    if free.size == 0:
        raise RuntimeError("No hay celdas libres para samplear.")
    idx = int(rng.integers(0, len(free)))
    r, c = free[idx]
    return int(r), int(c)


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])