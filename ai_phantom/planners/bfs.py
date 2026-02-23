from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai_phantom.envs.maze.maze_utils import in_bounds, neighbors_4

Pos = Tuple[int, int]


def bfs_plan(walls: np.ndarray, start: Pos, goal: Pos) -> Optional[List[Pos]]:
    """
    Devuelve la ruta (lista de posiciones) start->goal usando BFS en grid 4-dir.
    Si no hay ruta, retorna None.
    """
    h, w = walls.shape
    sr, sc = start
    gr, gc = goal
    if walls[sr, sc] or walls[gr, gc]:
        return None

    q = deque([start])
    parent: Dict[Pos, Optional[Pos]] = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in neighbors_4(r, c):
            if not in_bounds(nr, nc, h, w):
                continue
            if walls[nr, nc]:
                continue
            nxt = (nr, nc)
            if nxt in parent:
                continue
            parent[nxt] = (r, c)
            q.append(nxt)

    if goal not in parent:
        return None

    # reconstruir
    path: List[Pos] = []
    cur: Optional[Pos] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def path_to_actions(path: List[Pos]) -> List[int]:
    """
    Convierte una ruta de posiciones a acciones 0..3 (UP,DOWN,LEFT,RIGHT).
    """
    actions: List[int] = []
    for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
        dr, dc = r2 - r1, c2 - c1
        if dr == -1 and dc == 0:
            actions.append(0)
        elif dr == 1 and dc == 0:
            actions.append(1)
        elif dr == 0 and dc == -1:
            actions.append(2)
        elif dr == 0 and dc == 1:
            actions.append(3)
        else:
            raise ValueError(f"Paso inválido en path: {(r1,c1)} -> {(r2,c2)}")
    return actions