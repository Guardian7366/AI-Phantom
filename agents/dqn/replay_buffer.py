from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional, Any, Dict
from collections import deque

import numpy as np
import torch


@dataclass
class NStepTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class SumTree:
    """
    SumTree clásico:
      - tree[1] es la raíz (suma total)
      - hojas están en [capacity, 2*capacity)
      - data index = leaf_idx - capacity
    """
    def __init__(self, capacity: int):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"SumTree capacity debe ser > 0. Recibido: {capacity}")
        self.capacity = capacity

        # index 0 no se usa; root = 1
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.data: List[Optional[Any]] = [None] * self.capacity

        self.write = 0
        self.n_entries = 0

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float, item: Any):
        idx = self.write + self.capacity
        self.data[self.write] = item
        self.update(idx, float(priority))

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        idx = int(idx)
        priority = float(priority)

        change = priority - float(self.tree[idx])
        self.tree[idx] = priority

        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def get(self, s: float):
        """
        s en [0, total). (si llega == total por float, conviene clamp antes)
        """
        s = float(s)
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if s < float(self.tree[left]):
                idx = left
            else:
                s -= float(self.tree[left])
                idx = right

        data_idx = idx - self.capacity
        return int(idx), float(self.tree[idx]), self.data[data_idx]

    def reset(self):
        self.tree.fill(0.0)
        self.data = [None] * self.capacity
        self.write = 0
        self.n_entries = 0


class PrioritizedReplayBuffer:
    """
    PER + n-step.

    - Cada transición guardada incluye n_steps_real.
    - El agente debe usar gamma ** n_steps_real.

    sample() retorna:
      (obs_t, actions_t, rewards_t, next_obs_t, dones_t, n_steps_t, idxs, weights_t)
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200_000,
        eps: float = 1e-6,
        n_step: int = 3,
        gamma: float = 0.99,
        seed: int = 0,
        *,
        priority_clip: Optional[float] = None,  # None = desactivado
    ):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"capacity debe ser > 0. Recibido: {capacity}")

        self.capacity = capacity
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames) if int(beta_frames) > 0 else 1
        self.eps = float(eps) if float(eps) > 0 else 1e-6

        self.n_step = int(max(1, n_step))
        self.gamma = float(gamma)
        self.nstep_queue: Deque[NStepTransition] = deque(maxlen=self.n_step)

        self.tree = SumTree(self.capacity)

        # nuevas transiciones deben muestrearse; > 0
        self.max_priority = 1.0  # OJO: prioridad en el espacio almacenado en el árbol (ya sanitizada)
        self.frame = 1

        self.rng = np.random.default_rng(int(seed))

        # Clip opcional de prioridad para evitar explosiones (estabilidad lvl2)
        self.priority_clip = None if priority_clip is None else float(priority_clip)

    def reset(self):
        self.nstep_queue.clear()
        self.tree.reset()
        self.max_priority = 1.0
        self.frame = 1

    def __len__(self) -> int:
        return int(self.tree.n_entries)

    def beta(self) -> float:
        # anneal lineal hasta 1.0
        t = min(1.0, float(self.frame) / float(self.beta_frames))
        return float(self.beta_start + t * (1.0 - self.beta_start))

    def _pack_from_queue(self) -> Optional[Tuple[np.ndarray, int, float, np.ndarray, bool, int]]:
        """
        Empaqueta una transición n-step desde la cola actual.
        Retorna:
          (obs0, action0, R, next_obs_n, done_n, n_steps_real)
        """
        if len(self.nstep_queue) == 0:
            return None

        obs0 = self.nstep_queue[0].obs
        action0 = int(self.nstep_queue[0].action)

        R = 0.0
        done_n = False
        next_obs_n = self.nstep_queue[-1].next_obs

        n_real = 0
        for i, tr in enumerate(self.nstep_queue):
            R += (self.gamma ** i) * float(tr.reward)
            next_obs_n = tr.next_obs
            n_real = i + 1
            if tr.done:
                done_n = True
                break

        return obs0, action0, float(R), next_obs_n, bool(done_n), int(n_real)

    def _sanitize_priority(self, p: float) -> float:
        p = float(p)
        if (not np.isfinite(p)) or p <= 0.0:
            p = 1.0
        p = max(p, float(self.eps))
        if self.priority_clip is not None:
            p = min(p, float(self.priority_clip))
        return float(p)

    @staticmethod
    def _safe_copy_obs(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 3:
            arr = np.asarray(arr, dtype=np.float32).reshape(3, arr.shape[-2], arr.shape[-1])
        return np.ascontiguousarray(arr)

    def _add_packed(self, packed):
        p = self._sanitize_priority(self.max_priority)
        self.tree.add(p, packed)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        # ✅ blindaje: evita referencias mutables (barato para 3x8x8)
        obs_c = self._safe_copy_obs(obs)
        next_obs_c = self._safe_copy_obs(next_obs)

        self.nstep_queue.append(NStepTransition(obs_c, int(action), float(reward), next_obs_c, bool(done)))

        # Cuando está llena, añadimos 1 transición “normal”
        if len(self.nstep_queue) >= self.n_step:
            packed = self._pack_from_queue()
            if packed is not None:
                self._add_packed(packed)
            self.nstep_queue.popleft()

        # Si terminó el episodio, flush de lo que queda
        if bool(done):
            self._flush()

    def _flush(self):
        while len(self.nstep_queue) > 0:
            packed = self._pack_from_queue()
            if packed is not None:
                self._add_packed(packed)
            self.nstep_queue.popleft()

    def sample(self, batch_size: int, device: torch.device):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size debe ser > 0. Recibido: {batch_size}")

        self.frame += 1

        if len(self) == 0:
            raise RuntimeError("PER buffer vacío: no se puede samplear.")

        total = float(self.tree.total())
        if (not np.isfinite(total)) or total <= 0.0:
            raise RuntimeError(f"PER SumTree total inválido: total={total}")

        segment = total / float(batch_size)

        idxs: List[int] = []
        priorities: List[float] = []
        samples: List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []

        # Tiny para evitar s==total por flotantes
        tiny = 1e-8

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = float(self.rng.uniform(a, b))
            # ✅ clamp: evita caer exactamente en el borde (edge-case)
            if s >= total:
                s = float(total - tiny)
            if s < 0.0:
                s = 0.0

            idx, p, data = self.tree.get(s)

            if data is None or (not np.isfinite(p)) or p <= 0.0:
                # fallback global si sale basura
                total2 = float(self.tree.total())
                if (not np.isfinite(total2)) or total2 <= 0.0:
                    raise RuntimeError(f"PER SumTree total inválido en fallback: total={total2}")

                found = False
                for _retry in range(25):
                    s2 = float(self.rng.uniform(0.0, total2))
                    if s2 >= total2:
                        s2 = float(total2 - tiny)
                    idx, p, data = self.tree.get(s2)
                    if data is not None and np.isfinite(p) and p > 0.0:
                        found = True
                        break

                if not found:
                    raise RuntimeError("No se pudo samplear una transición válida del PER (hojas vacías / inválidas).")

            idxs.append(int(idx))
            priorities.append(float(p))
            samples.append(data)

        total = float(self.tree.total())
        probs = np.array(priorities, dtype=np.float32) / float(total + 1e-8)
        probs = np.clip(probs, 1e-8, 1.0)

        beta = float(self.beta())

        # IS weights
        weights = (float(len(self)) * probs) ** (-beta)
        weights /= (weights.max() + 1e-8)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
        obs, actions, rewards, next_obs, dones, n_steps = zip(*samples)

        # Ya están float32/contiguos por push(); stack es barato
        obs_np = np.stack(obs, axis=0)
        next_obs_np = np.stack(next_obs, axis=0)

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        next_obs_t = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

        actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        # n real por transición (para gamma**n_real)
        n_steps_t = torch.tensor(n_steps, dtype=torch.int64, device=device).unsqueeze(1)

        return (obs_t, actions_t, rewards_t, next_obs_t, dones_t, n_steps_t, idxs, weights_t)

    def update_priorities(self, idxs: List[int], priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)

        # td_error -> priority
        priorities = np.abs(priorities) + float(self.eps)
        priorities = np.nan_to_num(priorities, nan=1.0, posinf=1.0, neginf=1.0)
        if self.priority_clip is not None:
            priorities = np.minimum(priorities, float(self.priority_clip))
        priorities = priorities ** float(self.alpha)
        for idx, p in zip(idxs, priorities):
            p = self._sanitize_priority(float(p))
            self.tree.update(int(idx), p)
            self.max_priority = max(float(self.max_priority), float(p))

        # max_priority también debe respetar clip si existe
        if self.priority_clip is not None:
            self.max_priority = min(float(self.max_priority), float(self.priority_clip))

    # -------------------------
    # Reproducibilidad / sandbox
    # -------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Devuelve un estado para reproducir sampling/debug.

        NOTA: esto NO es JSON-serializable porque `data` contiene np.ndarray/tuplas.
        Es serializable vía pickle / torch.save (recomendado para sandbox reproducible).
        """
        return {
            "capacity": int(self.capacity),
            "alpha": float(self.alpha),
            "beta_start": float(self.beta_start),
            "beta_frames": int(self.beta_frames),
            "eps": float(self.eps),
            "n_step": int(self.n_step),
            "gamma": float(self.gamma),
            "max_priority": float(self.max_priority),
            "frame": int(self.frame),
            "write": int(self.tree.write),
            "n_entries": int(self.tree.n_entries),
            "tree": self.tree.tree.copy(),
            "data": list(self.tree.data),
            "rng_state": self.rng.bit_generator.state,
            "priority_clip": None if self.priority_clip is None else float(self.priority_clip),
            "nstep_queue_len": int(len(self.nstep_queue)),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restaura el estado de get_state().
        Defensivo: valida shapes y capacidad.
        """
        if int(state.get("capacity", -1)) != int(self.capacity):
            raise ValueError("set_state: capacity no coincide con este buffer.")

        self.alpha = float(state["alpha"])
        self.beta_start = float(state["beta_start"])
        self.beta_frames = int(state["beta_frames"])
        self.eps = float(state["eps"])
        self.n_step = int(state["n_step"])
        self.gamma = float(state["gamma"])

        self.max_priority = float(state["max_priority"])
        self.frame = int(state["frame"])

        self.priority_clip = state.get("priority_clip", None)
        if self.priority_clip is not None:
            self.priority_clip = float(self.priority_clip)

        tree_arr = np.asarray(state["tree"], dtype=np.float32)
        if tree_arr.shape != self.tree.tree.shape:
            raise ValueError("set_state: shape del SumTree.tree no coincide.")
        self.tree.tree[...] = tree_arr

        data_list = state["data"]
        if not isinstance(data_list, list) or len(data_list) != self.capacity:
            raise ValueError("set_state: data inválida o longitud incorrecta.")
        self.tree.data = list(data_list)

        self.tree.write = int(state["write"])
        self.tree.n_entries = int(state["n_entries"])

        rng_state = state["rng_state"]
        self.rng = np.random.default_rng(0)
        self.rng.bit_generator.state = rng_state

        # nstep_queue se deja vacía: estado de episodio en curso, no recomendado restaurarlo
        self.nstep_queue.clear()