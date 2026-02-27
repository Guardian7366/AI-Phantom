# utils/maze_game.py
from __future__ import annotations

import os
import math
import random
from typing import Optional, Tuple, Dict, Any, List

import pygame
import numpy as np
import torch

from ai_phantom.agents.ppo import CnnActorCritic, Policy
from ai_phantom.core import load_checkpoint, select_device
from ai_phantom.envs.maze.maze_env import MazeConfig, MazeEnv
from ai_phantom.envs.maze.maze_utils import bfs_distance_map  # BFS SOLO para reachability/distancia (NO control)
from utils.start_menu import Icon_Button, SettingsPanel
from utils.conf import WINDOW_WIDTH, FPS, Config


Pos = Tuple[int, int]


class MazeGameScreen:
    """
    Pantalla de inferencia (juego) del laberinto (UI).

    Reglas clave:
    - El entorno (MazeConfig) se adapta al checkpoint (phase/env_cfg/stage/wp/...).
    - El usuario coloca la "persona" (goal) con click, PERO:
      * debe ser alcanzable (BFS)
      * debe respetar un límite de distancia (dist_cap) para ser "justo"
    - El fantasma se mueve con PPO determinista (sin BFS fallback).

    Nota importante:
    - BFS se usa SOLO en placement para validar alcanzable/distancia.
      En running NO se llama BFS (contador BFS(running) debe ser 0).
    """

    # ----------------------
    # INIT
    # ----------------------
    def __init__(self, config: Config):
        # Config UI compartida
        self.clock = config.clock
        self.settings = config.settings
        self.screen = config.screen
        self.click_sound = config.click_sound
        self.caught_sound = config.caught_sound
        self.font_title = config.font_title
        self.font_statsTitle = config.font_statsTitle
        self.font_button = config.font_button
        self.font_text = config.font_text

        self.running = True
        self.show_settings = False

        # Control velocidad del fantasma
        self.current_tick = 0
        self.step_ms = 95  # ms entre pasos (ritmo del fantasma)

        # Estado del episodio (placing -> running -> done)
        self.state = "placing"
        self.done_cooldown_ms = 900
        self.done_tick = 0
        self.last_episode_result = ""

        # Hover / validación
        self.mouse_in_maze_pos: Optional[Pos] = None
        self.mouse_valid: bool = False
        self.mouse_reason: str = ""

        # Runtime
        self.current_action = "idle"
        self._obs: Optional[np.ndarray] = None
        self._info: Optional[Dict[str, Any]] = None
        self.player_pos: Optional[Pos] = None  # la “persona”/meta

        # Debug anti-sospecha (PPO-only)
        self._bfs_calls_placing = 0
        self._bfs_calls_running = 0  # debería quedarse en 0 siempre
        self._ppo_steps = 0

        # Cache para no spamear BFS al mover mouse dentro de la misma celda
        self._hover_cache_pos: Optional[Pos] = None
        self._hover_cache_ok: bool = False
        self._hover_cache_reason: str = ""

        # ---- UI / botones ----
        self._create_buttons()

        # Sprites (respetando assets existentes)
        self.ghost_img = {
            0: pygame.image.load("assets/sprites/phantom/PhantomBack.png").convert_alpha(),
            1: pygame.image.load("assets/sprites/phantom/PhantomFront.png").convert_alpha(),
            2: pygame.image.load("assets/sprites/phantom/PhantomLeft.png").convert_alpha(),
            3: pygame.image.load("assets/sprites/phantom/PhantomRight.png").convert_alpha(),
            "idle": pygame.image.load("assets/sprites/phantom/PhantomIdle.png").convert_alpha(),
        }
        self.floor_img = pygame.image.load("assets/sprites/misc/FloorMaze.png").convert()
        self.wall_img = pygame.image.load("assets/sprites/misc/Wall.png").convert()
        self.player_img = pygame.image.load("assets/sprites/misc/Player.png").convert_alpha()
        self.cross_img = pygame.image.load("assets/sprites/misc/RedCross.png").convert_alpha()

        # Hover overlay (verde/rojo)
        self._hover_overlay = pygame.Surface((1, 1), pygame.SRCALPHA)

        # Volúmenes
        self.settings.apply_music_volume()
        self.settings.apply_sfx_volume(self.click_sound)
        self.settings.apply_sfx_volume(self.caught_sound)

        self.settings_panel = SettingsPanel(self.screen, self.settings, self.click_sound, self.font_button)

        # ---- Checkpoint + device + env/model ----
        self.ckptpath = self._pick_checkpoint()
        if self.ckptpath is None:
            print("❌ No se encontró checkpoint (final/best).")
            self.running = False
            return

        dev_cfg = select_device(device="auto")
        self.device = dev_cfg.device  # target (normalmente cuda)

        self._load_model_and_env(self.ckptpath)

        # Layout final (ya con banner + rects correctos)
        self._recalc_layout()

        # Nuevo episodio
        self._new_episode()

    # ----------------------
    # CHECKPOINT PICK
    # ----------------------
    def _pick_checkpoint(self) -> Optional[str]:
        cand = [
            "results/checkpoints/final_phase1.pt",
            "results/checkpoints/best_phase1.pt",
            "results/checkpoints/best_phase0.pt",
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        return None

    # ----------------------
    # LOAD MODEL + ADAPT ENV
    # ----------------------
    def _load_model_and_env(self, ckpt_path: str) -> None:
        """
        Adapta la inferencia al checkpoint y evita terminar temprano por loops (solo en juego).
        """

        base_env_cfg: Dict[str, Any] = {
            "height": 12,
            "width": 12,
            "use_walls": True,
            "wall_prob": 0.18,
            "max_steps": 256,
            "min_manhattan": 6,
            "include_dist_channel": True,
            "include_goal": True,
            "include_visited": True,
            "include_step_channel": True,
            "enable_loop_detection": True,
            "terminate_on_loop": False,       # 👈 no “se rinde” por loops en UI
            "loop_window": 16,
            "stagnation_steps": 10,
            "loop_terminate_hits": 999999,
        }

        # Env temporal para inferir obs_shape
        tmp_cfg = MazeConfig(**base_env_cfg)
        tmp_env = MazeEnv(tmp_cfg, seed=0)
        obs0, _ = tmp_env.reset(seed=0, phase=1)
        obs_shape = tuple(obs0.shape)

        model = CnnActorCritic(obs_shape=obs_shape, num_actions=4)

        extra = load_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=None,
            map_location=self.device,
            restore_rng=False,
        )

        self.phase = int(extra.get("phase", 1 if "phase1" in ckpt_path else 0))

        env_cfg_from_ckpt = extra.get("env_cfg", None)
        if isinstance(env_cfg_from_ckpt, dict):
            base_env_cfg.update(env_cfg_from_ckpt)

        cur_wp = extra.get("cur_wall_prob", None)
        if cur_wp is not None:
            try:
                base_env_cfg["wall_prob"] = float(cur_wp)
            except Exception:
                pass

        if self.phase == 0:
            base_env_cfg["use_walls"] = False
            base_env_cfg["wall_prob"] = 0.0

        base_env_cfg["height"] = int(base_env_cfg.get("height", 12))
        base_env_cfg["width"] = int(base_env_cfg.get("width", 12))
        base_env_cfg["max_steps"] = int(base_env_cfg.get("max_steps", 256))
        base_env_cfg["min_manhattan"] = int(base_env_cfg.get("min_manhattan", 6))

        base_env_cfg["include_goal"] = True
        base_env_cfg.setdefault("include_visited", True)
        base_env_cfg.setdefault("include_step_channel", True)
        base_env_cfg.setdefault("include_dist_channel", True)

        # 👇 forzado en UI (aunque ckpt diga lo contrario)
        base_env_cfg["terminate_on_loop"] = False
        base_env_cfg["loop_terminate_hits"] = 999999

        self.maze_cfg = MazeConfig(**base_env_cfg)
        self.maze_env = MazeEnv(self.maze_cfg, seed=0)

        self.cur_stage = int(extra.get("cur_stage", -1))
        self.cur_wall_prob = float(getattr(self.maze_cfg, "wall_prob", 0.0))
        self.cur_min_manhattan = int(getattr(self.maze_cfg, "min_manhattan", 6))
        self.max_steps = int(getattr(self.maze_cfg, "max_steps", 256))

        self.model_sr = float(
            extra.get(
                "final_stop_sr_ppo",
                extra.get("best_test_sr_det_multiwalls_final", extra.get("best_eval_sr_det", -1.0)),
            )
        )

        self.dist_cap = self._compute_fair_dist_cap(
            sr=self.model_sr,
            min_manhattan=self.cur_min_manhattan,
            max_steps=self.max_steps,
        )

        try:
            model = model.to(self.device)
        except Exception as e:
            print(f"⚠️ No se pudo mover modelo a {self.device}. Usando CPU. Error: {e}")
            self.device = torch.device("cpu")
            model = model.to(self.device)

        model.eval()

        self.policy = Policy(
            model=model,
            enable_action_mask=True,
            nan_repl=0.0,
            fallback_action=0,
        )

        try:
            print(f"✅ Loaded: {ckpt_path}")
            print(f"   device_target: {self.device}")
            print(f"   model_device : {next(self.policy.model.parameters()).device}")
            print(
                f"   phase={self.phase} stage={self.cur_stage} wp={self.cur_wall_prob:.3f} "
                f"mh={self.cur_min_manhattan} max_steps={self.max_steps} dist_cap={self.dist_cap}"
            )
        except Exception:
            pass

    def _compute_fair_dist_cap(self, *, sr: float, min_manhattan: int, max_steps: int) -> int:
        mh = int(max(1, min_manhattan))
        ms = int(max(32, max_steps))

        if sr < 0.0:
            cap = 3 * mh + 10
        elif sr < 0.50:
            cap = 2 * mh + 10
        elif sr < 0.75:
            cap = 3 * mh + 12
        elif sr < 0.90:
            cap = 4 * mh + 14
        else:
            cap = 5 * mh + 16

        cap = int(max(12, min(cap, 64)))
        cap = int(min(cap, ms - 1))
        return cap

    # ----------------------
    # EPISODE CONTROL
    # ----------------------
    def _new_episode(self) -> None:
        self.state = "placing"
        self.player_pos = None
        self.last_episode_result = ""
        self.mouse_in_maze_pos = None
        self.mouse_valid = False
        self.mouse_reason = ""
        self.current_action = "idle"
        self._obs = None
        self._info = None

        self._bfs_calls_placing = 0
        self._bfs_calls_running = 0
        self._ppo_steps = 0

        self._hover_cache_pos = None
        self._hover_cache_ok = False
        self._hover_cache_reason = ""

        seed = random.randint(0, 999999)

        if getattr(self.maze_env, "rebuild_walls", None) is not None and bool(getattr(self.maze_cfg, "use_walls", False)):
            self.maze_env.rebuild_walls(seed=int(seed), wall_prob=float(self.cur_wall_prob))

        obs, info = self.maze_env.reset(seed=int(seed), phase=int(self.phase))
        self._obs = obs
        self._info = info

        self.maze_env.goal = None

    def _finish_episode(self, *, reached: bool, term_reason: str) -> None:
        self.state = "done"
        self.done_tick = pygame.time.get_ticks()

        if reached:
            self.last_episode_result = "CAUGHT ✅"
            if self.caught_sound:
                self.caught_sound.play()
        else:
            if len(term_reason) > 24:
                term_reason = term_reason[:24] + "..."
            self.last_episode_result = f"FAILED ({term_reason})"

    # ----------------------
    # UI: BUTTONS & LAYOUT
    # ----------------------
    def _create_buttons(self):
        self.btn_back = Icon_Button(
            (20, 20, 70, 70),
            "assets/images/back.png",
            self.font_button,
            (60, 60, 90),
            (90, 90, 140),
            click_sound=self.click_sound,
        )
        self.btn_settings = Icon_Button(
            (WINDOW_WIDTH - 160, 20, 75, 75),
            "assets/images/gear.png",
            self.font_button,
            (40, 40, 60),
            (80, 80, 120),
            click_sound=self.click_sound,
        )

    def _recalc_layout(self):
        width, height = self.screen.get_size()
        gap = 24

        self.title_pos = (width // 2, 48)

        self.banner_top = 86
        self.banner_h = 52
        self.banner_rect = pygame.Rect(0, self.banner_top, width, self.banner_h)

        area_top = self.banner_rect.bottom + 10
        area_bottom_pad = 70
        area_h = max(100, height - area_top - area_bottom_pad)

        stats_min_w = 260
        stats_w = max(stats_min_w, int(width * 0.22))
        stats_w = min(stats_w, width - 220)

        margin_left = max(30, int(width * 0.10))
        total_w = width - margin_left - 20
        maze_w = max(220, total_w - stats_w - gap)

        self.maze_rect = pygame.Rect(margin_left, area_top, maze_w, area_h)
        self.stats_rect = pygame.Rect(margin_left + maze_w + gap, area_top, stats_w, area_h)

        self.ghost_title_pos = (width // 4, 30)
        self.ghost_score_pos = (width // 4, 65)
        self.player_title_pos = (width * 0.75, 30)
        self.player_score_pos = (width * 0.75, 65)

        self.btn_back.rect.topleft = (20, 30)
        self.btn_back.rect.size = (70, 70)
        self.btn_settings.rect.topright = (width - 20, 20)
        self.btn_settings.rect.size = (75, 75)

        rows = int(getattr(self.maze_cfg, "height", 12))
        cols = int(getattr(self.maze_cfg, "width", 12))
        cell = min(self.maze_rect.width / cols, self.maze_rect.height / rows)
        grid_w = int(cell * cols)
        grid_h = int(cell * rows)

        self.grid_rect = pygame.Rect(0, 0, grid_w, grid_h)
        self.grid_rect.center = self.maze_rect.center

    # ----------------------
    # TEXT HELPERS (fit / wrap)
    # ----------------------
    def _safe_scale_surface(self, surf: pygame.Surface, size: Tuple[int, int], *, smooth: bool) -> pygame.Surface:
        """
        Evita el crash de pygame:
        - smoothscale SOLO acepta 24/32-bit.
        - Para texto/pixel art, scale normal es suficiente.
        """
        new_w, new_h = size
        new_w = max(1, int(new_w))
        new_h = max(1, int(new_h))

        if smooth:
            try:
                bpp = surf.get_bitsize()
                if bpp in (24, 32):
                    return pygame.transform.smoothscale(surf, (new_w, new_h))
            except Exception:
                pass

        # fallback seguro
        return pygame.transform.scale(surf, (new_w, new_h))

    def _render_fit(self, text: str, font: pygame.font.Font, color, max_w: int) -> pygame.Surface:
        """
        Renderiza texto y si no cabe, lo escala hacia abajo sin romper:
        - intenta smoothscale si es posible (24/32-bit)
        - si no, usa scale normal (seguro)
        """
        surf = font.render(text, False, color)

        # Intentar convertir a formato compatible (si display existe)
        try:
            # si la surface tiene alpha, convert_alpha suele dejarla a 32-bit
            surf = surf.convert_alpha()
        except Exception:
            # si el display no está listo aún, ignoramos
            pass

        w = surf.get_width()
        if w <= max_w or max_w <= 0:
            return surf

        scale = max(0.70, max_w / float(w))
        new_w = int(surf.get_width() * scale)
        new_h = int(surf.get_height() * scale)

        # Texto: preferimos NO suavizar (mejor para pixel fonts y evita blur)
        return self._safe_scale_surface(surf, (new_w, new_h), smooth=False)

    def _ellipsis_fit(self, text: str, font: pygame.font.Font, color, max_w: int) -> pygame.Surface:
        if max_w <= 0:
            return font.render("", False, color)

        s = text
        surf = font.render(s, False, color)
        try:
            surf = surf.convert_alpha()
        except Exception:
            pass

        if surf.get_width() <= max_w:
            return surf

        if len(s) <= 3:
            return self._render_fit(s, font, color, max_w)

        lo, hi = 0, len(s)
        best = "..."
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = s[:mid] + "..."
            cs = font.render(candidate, False, color)
            if cs.get_width() <= max_w:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        out = font.render(best, False, color)
        try:
            out = out.convert_alpha()
        except Exception:
            pass
        return out

    def _wrap_words(self, text: str, font: pygame.font.Font, max_w: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        cur: List[str] = []
        for w in words:
            test = (" ".join(cur + [w])).strip()
            if font.size(test)[0] <= max_w or not cur:
                cur.append(w)
            else:
                lines.append(" ".join(cur))
                cur = [w]
        if cur:
            lines.append(" ".join(cur))
        return lines

    # ----------------------
    # GRID HELPERS
    # ----------------------
    def _cell_size(self) -> Tuple[float, float]:
        cols = int(self.maze_cfg.width)
        rows = int(self.maze_cfg.height)
        return self.grid_rect.width / cols, self.grid_rect.height / rows

    def _mouse_to_cell(self, mouse_pos: Tuple[int, int]) -> Optional[Pos]:
        if not self.grid_rect.collidepoint(mouse_pos):
            return None
        rel_x = mouse_pos[0] - self.grid_rect.left
        rel_y = mouse_pos[1] - self.grid_rect.top
        cell_w, cell_h = self._cell_size()
        c = int(rel_x // cell_w)
        r = int(rel_y // cell_h)
        if 0 <= r < int(self.maze_cfg.height) and 0 <= c < int(self.maze_cfg.width):
            return (r, c)
        return None

    def _grid_char_at(self, r: int, c: int) -> str:
        if bool(self.maze_env.walls[r, c]):
            return "#"
        if (r, c) == tuple(self.maze_env.agent):
            return "A"
        if self.player_pos is not None and (r, c) == self.player_pos:
            return "G"
        return "."

    # ----------------------
    # DRAW
    # ----------------------
    def _draw_title(self):
        title = self.font_title.render("MAZE", False, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=self.title_pos))

    def _draw_banner(self) -> None:
        pygame.draw.rect(self.screen, (14, 16, 26), self.banner_rect)
        pygame.draw.line(self.screen, (80, 80, 90), (0, self.banner_rect.top), (self.banner_rect.width, self.banner_rect.top), 2)
        pygame.draw.line(self.screen, (50, 50, 60), (0, self.banner_rect.bottom), (self.banner_rect.width, self.banner_rect.bottom), 2)

        if self.state == "placing":
            msg = f"Click para colocar PERSONA (alcanzable) | Distancia <= {self.dist_cap}"
        elif self.state == "running":
            msg = "Fantasma buscando... (CONTROL: PPO determinista | NO BFS)"
        else:
            msg = f"{self.last_episode_result}  |  (click en el laberinto para nuevo intento)"

        max_w = self.banner_rect.width - 52
        lines = self._wrap_words(msg, self.font_text, max_w)[:2]
        y = self.banner_rect.top + 10
        for line in lines:
            surf = self._render_fit(line, self.font_text, (235, 235, 235), max_w)
            self.screen.blit(surf, (26, y))
            y += 18

    def _draw_stats_area(self) -> None:
        pygame.draw.rect(self.screen, (12, 14, 22), self.stats_rect)
        pygame.draw.rect(self.screen, (90, 90, 90), self.stats_rect, 3)

        x = self.stats_rect.left + 14
        y = self.stats_rect.top + 12
        max_y = self.stats_rect.bottom - 10
        max_w = self.stats_rect.width - 20

        def put_big(text: str):
            nonlocal y
            if y > max_y:
                return
            surf = self._render_fit(text, self.font_button, (235, 235, 235), max_w)
            self.screen.blit(surf, (x, y))
            y += int(surf.get_height() + 6)

        def put_line(text: str, *, ellipsis: bool = False):
            nonlocal y
            if y > max_y:
                return
            surf = self._ellipsis_fit(text, self.font_text, (235, 235, 235), max_w) if ellipsis else \
                   self._render_fit(text, self.font_text, (235, 235, 235), max_w)
            self.screen.blit(surf, (x, y))
            y += int(surf.get_height() + 4)

        put_big("INFO")
        put_line(f"CKPT: {os.path.basename(self.ckptpath)}", ellipsis=True)
        put_line("CONTROL: PPO (no BFS)")
        put_line(f"PPO SR: {self.model_sr:.3f}" if self.model_sr >= 0 else "PPO SR: unknown")
        put_line(f"PHASE: {self.phase}")
        if self.cur_stage >= 0:
            put_line(f"STAGE: {self.cur_stage}")
        put_line(f"WALL_PROB: {self.cur_wall_prob:.3f}")
        put_line(f"MIN_MANHATTAN: {self.cur_min_manhattan}")
        put_line(f"MAX_STEPS: {self.max_steps}")
        put_line(f"DIST_CAP: {self.dist_cap}")

        y += 6
        pygame.draw.line(self.screen, (70, 70, 80), (x, y), (self.stats_rect.right - 12, y), 2)
        y += 10

        put_line(f"STATE: {self.state.upper()}")
        put_line(f"PPO_STEPS: {self._ppo_steps}")
        put_line(f"BFS(placing): {self._bfs_calls_placing}")
        put_line(f"BFS(running): {self._bfs_calls_running} (debe ser 0)")

        if self.state == "placing" and self.mouse_in_maze_pos is not None:
            y += 6
            put_line(f"HOVER: {self.mouse_in_maze_pos} | {'OK' if self.mouse_valid else 'NO'}")
            if self.mouse_reason:
                put_line(f"WHY: {self.mouse_reason}", ellipsis=True)

    def _draw_maze_area(self):
        pygame.draw.rect(self.screen, (12, 14, 22), self.maze_rect)
        pygame.draw.rect(self.screen, (90, 90, 90), self.maze_rect, 3)
        pygame.draw.rect(self.screen, (150, 150, 150), self.maze_rect, 2)

        pygame.draw.rect(self.screen, (18, 20, 30), self.grid_rect)
        pygame.draw.rect(self.screen, (60, 60, 70), self.grid_rect, 2)

        rows = int(self.maze_cfg.height)
        cols = int(self.maze_cfg.width)
        cell_w, cell_h = self._cell_size()

        self._hover_overlay = pygame.Surface((max(1, int(cell_w)), max(1, int(cell_h))), pygame.SRCALPHA)
        if self.state == "placing" and self.mouse_in_maze_pos is not None:
            self._hover_overlay.fill((50, 220, 120, 70) if self.mouse_valid else (220, 60, 60, 70))

        for r in range(rows):
            for c in range(cols):
                x = int(self.grid_rect.left + c * cell_w)
                y = int(self.grid_rect.top + r * cell_h)
                rect = pygame.Rect(x, y, math.ceil(cell_w), math.ceil(cell_h))

                # pixel art friendly: scale normal
                self.screen.blit(pygame.transform.scale(self.floor_img, (rect.width, rect.height)), rect)

                ch = self._grid_char_at(r, c)
                if ch == "#":
                    self.screen.blit(pygame.transform.scale(self.wall_img, (rect.width, rect.height)), rect)
                elif ch == "G":
                    self.screen.blit(pygame.transform.scale(self.player_img, (rect.width, rect.height)), rect)
                elif ch == "A":
                    img_src = self.ghost_img.get(self.current_action, self.ghost_img["idle"])
                    self.screen.blit(pygame.transform.scale(img_src, (rect.width, rect.height)), rect)

                if self.state == "placing" and self.mouse_in_maze_pos == (r, c):
                    self.screen.blit(self._hover_overlay, rect.topleft)
                    self.screen.blit(pygame.transform.scale(self.cross_img, (rect.width, rect.height)), rect)

    def _draw_controls(self):
        mouse_pos = pygame.mouse.get_pos()
        for b in (self.btn_back, self.btn_settings):
            b.update(mouse_pos)
            b.draw(self.screen)

        steps_title = self.font_button.render("STEPS", False, (230, 230, 230))
        self.screen.blit(steps_title, steps_title.get_rect(center=self.ghost_title_pos))

        cur_steps = int(getattr(self.maze_env, "t", 0))
        cur_stepsLabel = self.font_button.render(str(cur_steps), False, (230, 230, 230))
        self.screen.blit(cur_stepsLabel, cur_stepsLabel.get_rect(center=self.ghost_score_pos))

        limit_title = self.font_button.render("LIMIT", False, (230, 230, 230))
        self.screen.blit(limit_title, limit_title.get_rect(center=self.player_title_pos))

        limitLabel = self.font_button.render(str(self.max_steps), False, (230, 230, 230))
        self.screen.blit(limitLabel, limitLabel.get_rect(center=self.player_score_pos))

    # ----------------------
    # INPUT VALIDATION (PLACEMENT)
    # ----------------------
    def _is_cell_placeable(self, pos: Pos) -> Tuple[bool, str]:
        if self._hover_cache_pos == pos:
            return self._hover_cache_ok, self._hover_cache_reason

        r, c = pos
        if r < 0 or r >= int(self.maze_cfg.height) or c < 0 or c >= int(self.maze_cfg.width):
            ok, reason = False, "fuera del grid"
        elif bool(self.maze_env.walls[r, c]):
            ok, reason = False, "es pared"
        elif (r, c) == tuple(self.maze_env.agent):
            ok, reason = False, "es el fantasma"
        else:
            self._bfs_calls_placing += 1
            dm = bfs_distance_map(self.maze_env.walls, (r, c))
            ar, ac = tuple(self.maze_env.agent)
            d = int(dm[ar, ac]) if dm is not None else -1

            if d < 0:
                ok, reason = False, "inalcanzable"
            elif d > int(self.dist_cap):
                ok, reason = False, f"muy lejos (dist={d})"
            else:
                ok, reason = True, f"ok (dist={d})"

        self._hover_cache_pos = pos
        self._hover_cache_ok = ok
        self._hover_cache_reason = reason
        return ok, reason

    def _commit_goal(self, pos: Pos) -> None:
        self.player_pos = pos
        self.maze_env.goal = tuple(pos)

        try:
            self._bfs_calls_placing += 1
            self.maze_env._dist_map = bfs_distance_map(self.maze_env.walls, tuple(pos))
        except Exception:
            pass

        try:
            self._obs = self.maze_env._make_obs()
        except Exception:
            pass

        self.state = "running"

    # ----------------------
    # PPO INFERENCE STEP
    # ----------------------
    def _obs_to_model_device(self, obs_np: np.ndarray) -> torch.Tensor:
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).float()
        try:
            model_dev = next(self.policy.model.parameters()).device
        except Exception:
            model_dev = torch.device("cpu")
        if obs_t.device != model_dev:
            obs_t = obs_t.to(model_dev)
        return obs_t

    def _update_ghost(self) -> None:
        if self.state != "running":
            return
        if self.player_pos is None or self.maze_env.goal is None:
            return

        if pygame.time.get_ticks() - self.current_tick <= self.step_ms:
            return
        self.current_tick = pygame.time.get_ticks()

        if self._obs is None:
            try:
                self._obs = self.maze_env._make_obs()
            except Exception:
                return

        obs_t = self._obs_to_model_device(self._obs)

        with torch.no_grad():
            out = self.policy.act(obs_t, deterministic=True)

        action = int(out.action.item())
        self.current_action = action
        self._ppo_steps += 1

        next_obs, reward, done, info = self.maze_env.step(action)
        self._obs = next_obs
        self._info = info

        if done:
            self.current_action = "idle"
            reached = bool(info.get("reached", False))
            term_reason = str(info.get("term_reason", "done"))
            self._finish_episode(reached=reached, term_reason=term_reason)

    # ----------------------
    # INTERACTION
    # ----------------------
    def _handle_control_click(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return None

        if self.btn_back.is_clicked(event):
            return "selection"

        if self.btn_settings.is_clicked(event):
            self.show_settings = True
            return None

        return None

    # ----------------------
    # MAIN LOOP
    # ----------------------
    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self._recalc_layout()

            if self.state == "done":
                if pygame.time.get_ticks() - self.done_tick > self.done_cooldown_ms:
                    self._new_episode()

            if self.state == "running":
                self._update_ghost()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                if self.show_settings:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        closed = self.settings_panel.handle_settings_click(event)
                        self.screen = pygame.display.get_surface()
                        self.settings_panel.screen = self.screen
                        if closed:
                            self.show_settings = False

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.show_settings = False
                    continue

                res = self._handle_control_click(event)
                if res == "selection":
                    return "selection"

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "selection"

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.maze_rect.collidepoint(event.pos) or self.stats_rect.collidepoint(event.pos)) and self.click_sound:
                        self.click_sound.play()

                if self.state in ("placing", "done") and event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                    cell = self._mouse_to_cell(pygame.mouse.get_pos())
                    if cell is None:
                        self.mouse_in_maze_pos = None
                        self.mouse_valid = False
                        self.mouse_reason = ""
                        self._hover_cache_pos = None
                    else:
                        self.mouse_in_maze_pos = cell
                        ok, reason = self._is_cell_placeable(cell)
                        self.mouse_valid = ok
                        self.mouse_reason = reason

                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if self.state == "done":
                                self._new_episode()
                            else:
                                if ok:
                                    self._commit_goal(cell)

            self.screen.fill((10, 12, 18))
            self._draw_title()
            self._draw_banner()
            self._draw_maze_area()
            self._draw_stats_area()
            self._draw_controls()

            if self.show_settings:
                overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (0, 0))
                self.settings_panel.draw_settings_panel()

            pygame.display.flip()

        return None