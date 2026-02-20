from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np


@dataclass
class StochasticConfig:
    """
    action_slip_prob:
      Probabilidad de reemplazar la acción ejecutada por otra (slip).
      0.0 => entorno determinista (wrapper neutro).

    slip_uniform:
      True => slip elige una acción aleatoria uniforme.

    slip_to_other:
      True => si ocurre slip, evita escoger la MISMA acción original.

    num_actions:
      Número de acciones discretas (por defecto 4 para Maze).
    """
    action_slip_prob: float = 0.0
    slip_uniform: bool = True
    slip_to_other: bool = True
    num_actions: int = 4


class StochasticWrapper:
    """
    Wrapper opcional para introducir estocasticidad SIN tocar el env base.
    Mantiene la misma API: reset(...) y step(action).

    Diseño importante:
    - El RNG del wrapper NO comparte el seed por episodio con el env,
      para evitar correlación entre slip y la generación del episodio.
    """

    def __init__(self, env, cfg: Optional[StochasticConfig] = None, seed: int = 0):
        self.env = env
        self.cfg = cfg or StochasticConfig()
        self.rng = np.random.default_rng(int(seed))

        # compatibilidad: algunos componentes esperan env.cfg.max_steps, etc.
        self._cfg_env = getattr(env, "cfg", None)

        self.last_action_slipped: bool = False
        self.last_action_before: Optional[int] = None
        self.last_action_after: Optional[int] = None

    def __getattr__(self, name: str):
        # passthrough al env base
        return getattr(self.env, name)

    @property
    def cfg_env(self):
        return self._cfg_env

    def seed(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        # NO reseed del wrapper con kwargs["seed"]
        self.last_action_slipped = False
        self.last_action_before = None
        self.last_action_after = None
        return self.env.reset(*args, **kwargs)

    def _slip_action(self, action: int) -> int:
        n = int(getattr(self.cfg, "num_actions", 4))
        if n <= 1:
            return int(action)

        # slip a otra acción distinta (si se pide)
        if bool(getattr(self.cfg, "slip_to_other", True)):
            a = int(action) % n
            r = int(self.rng.integers(0, n - 1))
            return r if r < a else (r + 1)

        return int(self.rng.integers(0, n))

    def step(self, action: int):
        a_in = int(action)
        a_out = a_in

        p = float(getattr(self.cfg, "action_slip_prob", 0.0))
        if p > 0.0 and float(self.rng.random()) < p:
            a_out = self._slip_action(a_in)
            self.last_action_slipped = True
        else:
            self.last_action_slipped = False

        self.last_action_before = a_in
        self.last_action_after = a_out

        obs, r, done, truncated, info = self.env.step(int(a_out))

        # normaliza info a dict
        if isinstance(info, dict):
            info_out = dict(info)
        else:
            info_out = {}

        info_out["action_slipped"] = bool(self.last_action_slipped)
        info_out["action_before"] = int(self.last_action_before)
        info_out["action_after"] = int(self.last_action_after)

        return obs, float(r), bool(done), bool(truncated), info_out