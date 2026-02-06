"""
Maze State module
-----------------
Define el estado observable del agente dentro del laberinto.
Este módulo NO contiene lógica de entrenamiento ni del entorno;
solo encapsula y valida la representación del estado.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class MazeState:
    """
    Representa el estado observable del agente en el laberinto.

    Componentes (alineados con el diseño conceptual):
    - agent_pos: posición (x, y) del fantasma
    - goal_pos: posición (x, y) del objetivo
    - walls: información local de paredes (arriba, abajo, izquierda, derecha)
    - dist_to_goal: distancia Manhattan al objetivo
    """

    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    walls: Tuple[int, int, int, int]  # (up, down, left, right) -> 1 si hay pared
    dist_to_goal: int

    # --------------------------------------------------
    # Validación básica
    # --------------------------------------------------
    def __post_init__(self):
        if len(self.walls) != 4:
            raise ValueError("walls debe tener exactamente 4 valores")

    # --------------------------------------------------
    # Representación numérica
    # --------------------------------------------------
    def to_vector(self, normalize: bool = False, maze_size: Tuple[int, int] | None = None) -> np.ndarray:
        """
        Convierte el estado a un vector numérico para el DQN.

        normalize:
            Si True, normaliza posiciones y distancia según maze_size.
        maze_size:
            Tamaño del laberinto (width, height), requerido si normalize=True.
        """
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        vector = [
            ax, ay,
            gx, gy,
            *self.walls,
            self.dist_to_goal,
        ]

        if normalize:
            if maze_size is None:
                raise ValueError("maze_size es requerido para normalizar")

            w, h = maze_size
            vector = [
                ax / w,
                ay / h,
                gx / w,
                gy / h,
                *self.walls,
                self.dist_to_goal / (w + h),
            ]

        return np.array(vector, dtype=np.float32)

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------
    def relative_goal_direction(self) -> Tuple[int, int]:
        """
        Dirección relativa del objetivo respecto al agente.
        Devuelve valores en {-1, 0, 1} para (dx, dy).
        """
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        dx = int(np.sign(gx - ax))
        dy = int(np.sign(gy - ay))

        return dx, dy
