from typing import Dict, Any
import numpy as np
import torch


class MazeState:
    """
    Wrapper de estado que traduce hechos crudos del entorno (maze_env)
    a una representación numérica apta para Deep Learning.

    El agente SOLO interactúa con esta clase.
    """

    def __init__(self, maze_width: int, maze_height: int):
        self.width = maze_width
        self.height = maze_height

        # Memoria interna
        self.agent_pos = (0, 0)
        self.goal_pos = None
        self.walls = np.zeros((maze_height, maze_width), dtype=np.float32)
        self.visited = np.zeros((maze_height, maze_width), dtype=np.float32)

        self.done = False
        self.last_raw_state: Dict[str, Any] | None = None

    # --------------------------------------------------
    # Entrada: hechos del entorno
    # --------------------------------------------------

    def update_from_facts(self, raw_state: Dict[str, Any]) -> None:
        """
        Consume el diccionario de hechos emitido por maze_env
        y actualiza el estado interno del wrapper.
        """
        self.last_raw_state = raw_state

        # Posición del agente
        x = raw_state["agent"]["x"]
        y = raw_state["agent"]["y"]
        self.agent_pos = (x, y)

        # Marcar visita
        self.visited[x, y] = 1.0

        # Terminal
        self.done = raw_state["terminal"]["done"]

        # Tipo de celda actual
        cell_type = raw_state["cell"]["type"]

        # Inferir goal si se detecta
        if cell_type == "goal":
            self.goal_pos = (x, y)

        # Inicializar muros si aún no están marcados
        # (Esto asume que el mapa completo se irá revelando)
        if cell_type == "wall":
            self.walls[x, y] = 1.0

    # --------------------------------------------------
    # Salida: representación para el agente
    # --------------------------------------------------

    def to_tensor(self) -> torch.Tensor:
        """
        Convierte el estado actual a un tensor normalizado
        con forma (C, H, W).
        """
        agent_layer = np.zeros((self.height, self.width), dtype=np.float32)
        agent_layer[self.agent_pos[0], self.agent_pos[1]] = 1.0

        goal_layer = np.zeros((self.height, self.width), dtype=np.float32)
        if self.goal_pos is not None:
            goal_layer[self.goal_pos[0], self.goal_pos[1]] = 1.0

        # Stack de canales
        state_tensor = np.stack(
            [
                self.walls,     # Canal 0: muros
                agent_layer,    # Canal 1: agente
                goal_layer,     # Canal 2: objetivo
                self.visited    # Canal 3: visitas
            ],
            axis=0
        )

        return torch.from_numpy(state_tensor)

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------

    def reset(self) -> None:
        """Reinicia completamente el estado."""
        self.agent_pos = (0, 0)
        self.goal_pos = None
        self.walls.fill(0.0)
        self.visited.fill(0.0)
        self.done = False
        self.last_raw_state = None
