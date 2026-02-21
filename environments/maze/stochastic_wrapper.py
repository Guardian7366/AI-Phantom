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
      True  => slip elige una acción aleatoria uniforme (default).
      False => slip usa un mapeo determinista "rotación" (a+1 mod n) para tests/sandbox.

    slip_to_other:
      True => si ocurre slip, evita escoger la MISMA acción original.

    num_actions:
      Número de acciones discretas (por defecto 4 para Maze).

    reseed_on_env_reset:
      False (default): NO reseed con el seed del env (evita correlación).
      True: reseed por episodio para reproducibilidad estricta del slip.

    reseed_salt:
      Offset/mezcla para derivar un seed del wrapper por episodio sin igualarlo al del env.
    """
    action_slip_prob: float = 0.0
    slip_uniform: bool = True
    slip_to_other: bool = True
    num_actions: int = 4

    reseed_on_env_reset: bool = False
    reseed_salt: int = 17_777


class StochasticWrapper:
    """
    Wrapper opcional para introducir estocasticidad SIN tocar el env base.
    Mantiene la misma API: reset(...) y step(action).

    Diseño:
    - Por defecto (reseed_on_env_reset=False): el RNG del wrapper NO se reseedea con el seed del episodio.
      Esto evita correlación entre slip y la generación del episodio (bueno para training).
    - Para sandbox/eval reproducible: reseed_on_env_reset=True hace slip determinista por episodio
      usando una derivación del seed del env + reseed_salt (independiente, pero reproducible).
    """

    def __init__(self, env, cfg: Optional[StochasticConfig] = None, seed: int = 0):
        self.env = env
        self.stoch_cfg = cfg or StochasticConfig()
        self.rng = np.random.default_rng(int(seed))

        # Compatibilidad: muchos componentes esperan env.cfg
        self.cfg = getattr(env, "cfg", None)

        self.last_action_slipped: bool = False
        self.last_action_before: Optional[int] = None
        self.last_action_after: Optional[int] = None

        # Telemetría barata
        self.total_steps: int = 0
        self.total_slips: int = 0

        # Seed base del wrapper (útil para reproducibilidad al restaurar estado)
        self._base_seed: int = int(seed)

    def __getattr__(self, name: str):
        # passthrough al env base (grid, agent_pos, goal_pos, etc.)
        return getattr(self.env, name)

    def seed(self, seed: int):
        """Reseedea SOLO el wrapper (slip). No afecta al env base."""
        self._base_seed = int(seed)
        self.rng = np.random.default_rng(int(seed))

    def _maybe_reseed_on_reset(self, kwargs: Dict[str, Any]) -> None:
        """
        Si está activado reseed_on_env_reset y viene un 'seed' en reset(), reseedea el wrapper
        con un seed derivado para mantener independencia pero reproducibilidad total.
        """
        if not bool(getattr(self.stoch_cfg, "reseed_on_env_reset", False)):
            return

        if "seed" not in kwargs or kwargs["seed"] is None:
            return

        env_seed = int(kwargs["seed"])
        salt = int(getattr(self.stoch_cfg, "reseed_salt", 17_777))

        # Derivación simple y estable (evita que sea igual al env_seed)
        # (env_seed XOR salt) + base_seed mezcla mejor sin costo.
        derived = (env_seed ^ salt) + (self._base_seed * 9973)

        self.rng = np.random.default_rng(int(derived))

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Decide si reseed (solo si config lo pide)
        # Importante: NO reseed por default (tu intención original se conserva).
        try:
            self._maybe_reseed_on_reset(kwargs if isinstance(kwargs, dict) else {})
        except Exception:
            # ultra defensivo: jamás romper reset()
            pass

        self.last_action_slipped = False
        self.last_action_before = None
        self.last_action_after = None

        # Reset telemetría por episodio (opcional; útil para sandbox)
        self.total_steps = 0
        self.total_slips = 0

        obs, info = self.env.reset(*args, **kwargs)
        info_out = dict(info) if isinstance(info, dict) else {}

        # Reporta config efectiva (útil para trazabilidad en eval)
        info_out["stoch_action_slip_prob"] = float(getattr(self.stoch_cfg, "action_slip_prob", 0.0))
        info_out["stoch_reseed_on_env_reset"] = bool(getattr(self.stoch_cfg, "reseed_on_env_reset", False))

        return obs, info_out

    def _slip_action(self, action: int) -> int:
        n = int(getattr(self.stoch_cfg, "num_actions", 4))
        if n <= 1:
            return int(action)

        a = int(action) % n

        # Modo NO-uniforme: determinista (ideal para pruebas y sandbox)
        if not bool(getattr(self.stoch_cfg, "slip_uniform", True)):
            # rotación simple (a+1) y si slip_to_other=False podría devolverse a (pero eso sería raro)
            cand = (a + 1) % n
            if bool(getattr(self.stoch_cfg, "slip_to_other", True)):
                return int(cand)
            return int(cand)

        # Modo uniforme
        if bool(getattr(self.stoch_cfg, "slip_to_other", True)):
            # elige una acción distinta sin sesgo
            r = int(self.rng.integers(0, n - 1))
            return int(r if r < a else (r + 1))

        return int(self.rng.integers(0, n))

    def step(self, action: int):
        a_in = int(action)
        a_out = a_in

        p = float(getattr(self.stoch_cfg, "action_slip_prob", 0.0))
        if p > 0.0 and float(self.rng.random()) < p:
            a_out = self._slip_action(a_in)
            self.last_action_slipped = True
            self.total_slips += 1
        else:
            self.last_action_slipped = False

        self.last_action_before = a_in
        self.last_action_after = a_out
        self.total_steps += 1

        obs, r, done, truncated, info = self.env.step(int(a_out))

        # normaliza info a dict
        info_out = dict(info) if isinstance(info, dict) else {}

        # Telemetría alineada (Ley 2)
        info_out["action_slipped"] = bool(self.last_action_slipped)
        info_out["action_before"] = int(self.last_action_before)
        info_out["action_after"] = int(self.last_action_after)
        info_out["stoch_slips_so_far"] = int(self.total_slips)
        info_out["stoch_steps_so_far"] = int(self.total_steps)
        info_out["stoch_action_slip_prob"] = float(p)

        return obs, float(r), bool(done), bool(truncated), info_out

    # -------------------------
    # Reproducibilidad / sandbox
    # -------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "base_seed": int(self._base_seed),
            "rng_state": self.rng.bit_generator.state,
            "total_steps": int(self.total_steps),
            "total_slips": int(self.total_slips),
            "cfg": {
                "action_slip_prob": float(getattr(self.stoch_cfg, "action_slip_prob", 0.0)),
                "slip_uniform": bool(getattr(self.stoch_cfg, "slip_uniform", True)),
                "slip_to_other": bool(getattr(self.stoch_cfg, "slip_to_other", True)),
                "num_actions": int(getattr(self.stoch_cfg, "num_actions", 4)),
                "reseed_on_env_reset": bool(getattr(self.stoch_cfg, "reseed_on_env_reset", False)),
                "reseed_salt": int(getattr(self.stoch_cfg, "reseed_salt", 17_777)),
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._base_seed = int(state.get("base_seed", 0))

        # Restaura cfg sin obligarte a reconstruir el wrapper
        cfg = state.get("cfg", {})
        self.stoch_cfg.action_slip_prob = float(cfg.get("action_slip_prob", self.stoch_cfg.action_slip_prob))
        self.stoch_cfg.slip_uniform = bool(cfg.get("slip_uniform", self.stoch_cfg.slip_uniform))
        self.stoch_cfg.slip_to_other = bool(cfg.get("slip_to_other", self.stoch_cfg.slip_to_other))
        self.stoch_cfg.num_actions = int(cfg.get("num_actions", self.stoch_cfg.num_actions))
        self.stoch_cfg.reseed_on_env_reset = bool(cfg.get("reseed_on_env_reset", self.stoch_cfg.reseed_on_env_reset))
        self.stoch_cfg.reseed_salt = int(cfg.get("reseed_salt", self.stoch_cfg.reseed_salt))

        self.total_steps = int(state.get("total_steps", 0))
        self.total_slips = int(state.get("total_slips", 0))

        rng_state = state.get("rng_state", None)
        self.rng = np.random.default_rng(0)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state