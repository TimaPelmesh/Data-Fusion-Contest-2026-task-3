#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import random
import argparse
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# DPI awareness
# ------------------------------------------------------------
if os.name == "nt":
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import pygame


# ============================================================
# Глобальные константы
# ============================================================

VISIT_COST = 100
HERO_COST = 2500
DAYS = 7
UI_SCALE = 1.30

PANEL_TITLE_SIZE = 16

HELP_TITLE_SIZE = 24
HELP_SECTION_SIZE = 18
HELP_BODY_SIZE = 17
HELP_HINT_SIZE = 13

TOOLBAR_BUTTON_SIZE = 13
TOOLBAR_SYMBOL_SIZE = 17
TOOLBAR_SPEED_SIZE = 13

INFO_LABEL_SIZE = 13
INFO_VALUE_SIZE = 13

LOG_FONT_SIZE = 14
ROUTES_FONT_SIZE = 14
ROUTES_SMALL_SIZE = 13

TOOLTIP_FONT_SIZE = 13
LEGEND_FONT_SIZE = 12

SCROLLBAR_W = 12
SCROLLBAR_GAP = 8
SCROLLBAR_MIN_THUMB_H = 26

MOUSE_WHEEL_STEP = 60
ROUTE_ARROW = " -> "
UNDO_HISTORY_LIMIT = 5000


# ============================================================
# Теплая тёмная тема
# ============================================================

BG = (24, 20, 18)
PANEL = (39, 33, 29)
PANEL_ALT = (46, 38, 33)
BORDER = (103, 88, 77)

TEXT = (240, 231, 218)
MUTED = (189, 171, 156)
GRID = (63, 54, 48)

TOOLBAR = (34, 28, 25)
BUTTON_FILL = (56, 46, 40)
BUTTON_HOVER = (72, 58, 49)
BUTTON_ACTIVE = (100, 74, 55)
BUTTON_BORDER = (120, 95, 76)
BUTTON_DISABLED = (45, 40, 37)
BUTTON_DISABLED_TEXT = (134, 122, 113)

LABEL_ACCENT = (236, 187, 121)
VALUE_ACCENT = TEXT
LEGEND_TEXT = (172, 160, 150)

DEPOT_LABEL = (245, 225, 189)
TAVERN_WOOD = (164, 118, 74)
TAVERN_WOOD_DARK = (118, 83, 51)
TAVERN_ROOF = (126, 58, 44)
TAVERN_DOOR = (78, 50, 31)

ANCHOR = (217, 185, 117)
UNASSIGNED = (122, 112, 104)
REMOVED = (255, 120, 64)
INSERTED = (94, 205, 255)
ACCEPT = (124, 210, 126)
REJECT = (236, 112, 112)
IN_PROGRESS = (230, 193, 96)
HOVER_RING = (222, 135, 255)
ROUTE_HOVER = (255, 244, 221)

RELATED_ROUTE = (75, 56, 49)
ROW_HOVER = (86, 67, 58)

SCROLL_TRACK = (52, 43, 38)
SCROLL_THUMB = (131, 106, 87)
SCROLL_THUMB_HOVER = (158, 127, 101)

TOOLTIP_BG = (249, 243, 233)
TOOLTIP_TEXT = (34, 28, 24)
TOOLTIP_BORDER = (138, 120, 103)

HELP_OVERLAY = (10, 8, 7, 195)
HELP_TEXT = (246, 238, 228)
HELP_HINT = (223, 206, 188)

HERO_COLORS = [
    (110, 164, 255),
    (255, 178, 102),
    (141, 216, 124),
    (212, 144, 228),
    (128, 205, 255),
    (255, 220, 114),
    (255, 138, 161),
    (116, 225, 172),
    (154, 184, 255),
    (242, 136, 126),
    (110, 205, 145),
    (177, 146, 255),
    (126, 184, 225),
    (238, 201, 106),
    (231, 109, 188),
    (166, 204, 255),
]

PARCHMENT_BASE = (86, 70, 52)
PARCHMENT_MID = (98, 80, 60)
PARCHMENT_LIGHT = (122, 102, 76)
PARCHMENT_DARK = (56, 44, 31)
PARCHMENT_STAIN = (36, 27, 19)
PARCHMENT_FIBER = (150, 129, 98)

PARCHMENT_BASE = (62, 46, 33)
PARCHMENT_MID = (74, 55, 39)
PARCHMENT_LIGHT = (92, 70, 51)
PARCHMENT_DARK = (48, 35, 25)
PARCHMENT_STAIN = (32, 23, 16)
PARCHMENT_FIBER = (112, 86, 63)

WOOD_LIGHT = (139, 102, 66)
WOOD_MID = (101, 71, 45)
WOOD_DARK = (62, 41, 24)

BRONZE_LIGHT = (198, 159, 101)
BRONZE = (150, 110, 68)
BRONZE_DARK = (95, 68, 42)

MAP_BASE = (88, 99, 72)
MAP_GRASS_1 = (102, 116, 80)
MAP_GRASS_2 = (77, 89, 63)
MAP_DIRT = (121, 105, 80)
MAP_HILL = (68, 76, 56)
MAP_RIVER = (73, 94, 108)
MAP_FOREST = (63, 84, 54)

PANEL_SHADOW = (0, 0, 0, 90)

# ============================================================
# Вспомогательные функции
# ============================================================

def now_sec() -> float:
    return time.perf_counter()


def wall_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_console(msg: str) -> None:
    print(f"{wall_timestamp()} | {msg}", flush=True)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mix_color(c1, c2, t: float):
    t = clamp(float(t), 0.0, 1.0)
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def nudge_saturation(rgb: Tuple[int, int, int], amount: float = 0.10) -> Tuple[int, int, int]:
    r, g, b = rgb
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return (
        int(clamp(gray + (r - gray) * (1.0 + amount), 0, 255)),
        int(clamp(gray + (g - gray) * (1.0 + amount), 0, 255)),
        int(clamp(gray + (b - gray) * (1.0 + amount), 0, 255)),
    )


def rgba(rgb, alpha: int):
    return (rgb[0], rgb[1], rgb[2], int(alpha))


def better_key(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    if a[0] != b[0]:
        return a[0] > b[0]
    return a[1] > b[1]


def parse_bool_flag(x: str) -> bool:
    x = x.strip().lower()
    return x in {"1", "true", "yes", "y", "on"}


def ext_distance_from_full(full, ext_a: int, ext_b: int) -> int:
    if ext_a == 0 and ext_b == 0:
        return 0
    if ext_a == 0:
        return int(full.dist_start_by_objid[ext_b])
    if ext_b == 0:
        return int(full.dist_start_by_objid[ext_a])
    return full.dist_by_objid(ext_a, ext_b)


def format_ext_name(ext: int) -> str:
    return "Таверна" if ext == 0 else str(ext)


def compact_ext_name(ext: int) -> str:
    return "Т" if ext == 0 else str(ext)


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    seg_len2 = vx * vx + vy * vy
    if seg_len2 <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = (wx * vx + wy * vy) / seg_len2
    t = clamp(t, 0.0, 1.0)
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return math.hypot(px - proj_x, py - proj_y)


def draw_alpha_rounded_rect(surface: pygame.Surface, rgba: Tuple[int, int, int, int], rect: pygame.Rect, radius: int = 4) -> None:
    tmp = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(tmp, rgba, tmp.get_rect(), border_radius=radius)
    surface.blit(tmp, rect.topleft)


def steps_text(x: int) -> str:
    return f"{int(x)} шагов"


# ============================================================
# Конфигурация приложения
# ============================================================

@dataclass
class AppConfig:
    data_dir: Path
    day: int = 1
    heroes: int = 6
    objects: int = 100
    sample_mode: str = "mixed"
    seed: int = 42
    texture_seed: int = 42
    warmup_previous: bool = False
    warmup_max_objects: int = 120
    iterations: int = 150
    rcl_size: int = 5
    destroy_frac_min: float = 0.10
    destroy_frac_max: float = 0.35
    temp_start: float = 0.20
    temp_end: float = 0.001
    width: int = 1600
    height: int = 960
    fps: int = 30
    steps_per_sec: int = 2
    supersampling: float = 1.0


def parse_args() -> AppConfig:
    default_data_dir = Path(__file__).absolute().parent / 'data'

    parser = argparse.ArgumentParser(
        description="Визуализатор LNS для задачи маршрутизации героев (Data Fusion 2026 - Heroes Task)"
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--day", type=int, default=1)
    parser.add_argument("--heroes", type=int, default=6)
    parser.add_argument("--objects", type=int, default=100)
    parser.add_argument("--sample-mode", type=str, default="mixed",
                        choices=["nearest", "random", "mixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-previous", type=str, default="false")
    parser.add_argument("--warmup-max-objects", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--rcl-size", type=int, default=5)
    parser.add_argument("--destroy-frac-min", type=float, default=0.10)
    parser.add_argument("--destroy-frac-max", type=float, default=0.35)
    parser.add_argument("--temp-start", type=float, default=0.20)
    parser.add_argument("--temp-end", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--steps-per-sec", type=int, default=2)
    parser.add_argument("--supersampling", type=float, default=1.0,
        help="Внутренний рендер в N раз больше окна с последующим smoothscale обратно (1.0 = выкл.)")
    args = parser.parse_args()

    cfg = AppConfig(
        data_dir=args.data_dir.resolve(),
        day=args.day,
        heroes=args.heroes,
        objects=args.objects,
        sample_mode=args.sample_mode,
        seed=args.seed,
        warmup_previous=parse_bool_flag(args.warmup_previous),
        warmup_max_objects=args.warmup_max_objects,
        iterations=args.iterations,
        rcl_size=args.rcl_size,
        destroy_frac_min=args.destroy_frac_min,
        destroy_frac_max=args.destroy_frac_max,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        width=args.width,
        height=args.height,
        fps=int(clamp(args.fps, 1, 120)),
        steps_per_sec=max(1, int(args.steps_per_sec)),
        supersampling=float(args.supersampling),
    )

    if not (1 <= cfg.day <= DAYS):
        raise RuntimeError("--day должен быть от 1 до 7")
    if cfg.heroes <= 0:
        raise RuntimeError("--heroes должно быть > 0")
    if cfg.objects < 0:
        raise RuntimeError("--objects должно быть >= 0")
    if cfg.iterations < 0:
        raise RuntimeError("--iterations должно быть >= 0")
    if cfg.rcl_size <= 0:
        raise RuntimeError("--rcl-size должно быть > 0")
    if cfg.destroy_frac_min <= 0 or cfg.destroy_frac_max <= 0 or cfg.destroy_frac_min > cfg.destroy_frac_max:
        raise RuntimeError("Некорректные destroy-frac параметры")
    if cfg.width < 980 or cfg.height < 650:
        raise RuntimeError("Окно слишком маленькое")
    if cfg.steps_per_sec <= 0:
        raise RuntimeError("--steps-per-sec должно быть > 0")
    if not (1.0 <= cfg.supersampling <= 4.0):
        raise RuntimeError("--supersampling должен быть в диапазоне [1.0, 4.0]")

    return cfg


# ============================================================
# Данные задачи
# ============================================================

@dataclass
class HeroState:
    anchor_ext: int = 0
    carry_discount: int = 0


@dataclass
class FullData:
    hero_caps: List[int]
    full_object_count: int
    object_day_open: np.ndarray
    dist_start_by_objid: np.ndarray
    dist_full: np.ndarray

    def hero_count(self) -> int:
        return len(self.hero_caps)

    def dist_by_objid(self, obj_a: int, obj_b: int) -> int:
        return int(self.dist_full[obj_a - 1, obj_b - 1])

    @staticmethod
    def load(data_dir: Path) -> "FullData":
        heroes_path = data_dir / "data_heroes.csv"
        objects_path = data_dir / "data_objects.csv"
        dist_start_path = data_dir / "dist_start.csv"
        dist_matrix_path = data_dir / "dist_objects.csv"

        log_console(f"Загрузка данных из: {data_dir}")
        t0 = now_sec()

        heroes_df = pd.read_csv(heroes_path, dtype={"hero_id": np.int32, "move_points": np.int32})
        heroes_df = heroes_df.sort_values("hero_id")
        hero_caps = heroes_df["move_points"].astype(np.int32).tolist()

        objects_df = pd.read_csv(
            objects_path,
            dtype={"object_id": np.int32, "day_open": np.int16, "reward": np.int32},
        )
        max_obj_id = int(objects_df["object_id"].max())

        object_day_open = np.zeros(max_obj_id + 1, dtype=np.int16)
        obj_ids = objects_df["object_id"].to_numpy(dtype=np.int32)
        day_vals = objects_df["day_open"].to_numpy(dtype=np.int16)
        object_day_open[obj_ids] = day_vals

        dist_start_df = pd.read_csv(
            dist_start_path,
            dtype={"object_id": np.int32, "dist_start": np.int32},
        )
        dist_start_by_objid = np.zeros(max_obj_id + 1, dtype=np.int32)
        ds_ids = dist_start_df["object_id"].to_numpy(dtype=np.int32)
        ds_vals = dist_start_df["dist_start"].to_numpy(dtype=np.int32)
        dist_start_by_objid[ds_ids] = ds_vals

        dist_full = pd.read_csv(dist_matrix_path, dtype=np.int32).to_numpy(copy=True)
        if dist_full.shape != (max_obj_id, max_obj_id):
            raise RuntimeError(
                f"Неверный размер матрицы: {dist_full.shape}, ожидалось {(max_obj_id, max_obj_id)}"
            )

        log_console(f"Героев: {len(hero_caps)}")
        log_console(f"Мельниц: {max_obj_id}")
        log_console(f"Данные загружены за {now_sec() - t0:.2f} сек")

        return FullData(
            hero_caps=hero_caps,
            full_object_count=max_obj_id,
            object_day_open=object_day_open,
            dist_start_by_objid=dist_start_by_objid,
            dist_full=dist_full,
        )


@dataclass
class DayData:
    day: int = 1
    num_heroes: int = 0
    object_count: int = 0
    hero_caps: List[int] = field(default_factory=list)
    object_ids_ext: List[int] = field(default_factory=list)
    ext_to_int: Dict[int, int] = field(default_factory=dict)
    start_cost_flat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    dist_flat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))

    def dist(self, a: int, b: int) -> int:
        return int(self.dist_flat[a, b])

    def start_cost(self, hero_idx: int, obj_idx: int) -> int:
        return int(self.start_cost_flat[hero_idx, obj_idx])

    def hero_capacity(self, h: int) -> int:
        return int(self.hero_caps[h])

    def object_id(self, obj: int) -> int:
        return int(self.object_ids_ext[obj])

    @staticmethod
    def build_for_day(
        full: FullData,
        day: int,
        heroes_count: int,
        hero_states_before_day: List[HeroState],
        selected_object_ids_ext: Optional[List[int]] = None,
    ) -> "DayData":
        if heroes_count > full.hero_count():
            raise RuntimeError("Запрошено героев больше, чем доступно")
        if len(hero_states_before_day) != heroes_count:
            raise RuntimeError("hero_states_before_day.size() != heroes_count")

        data = DayData()
        data.day = day
        data.num_heroes = heroes_count
        data.hero_caps = full.hero_caps[:heroes_count]

        if selected_object_ids_ext is None:
            object_ids_ext_arr = (np.where(full.object_day_open[1:] == day)[0] + 1).astype(np.int32)
            data.object_ids_ext = object_ids_ext_arr.tolist()
        else:
            data.object_ids_ext = sorted(int(x) for x in selected_object_ids_ext)
            object_ids_ext_arr = np.array(data.object_ids_ext, dtype=np.int32)

        data.ext_to_int = {ext: i for i, ext in enumerate(data.object_ids_ext)}
        data.object_count = len(data.object_ids_ext)

        if data.object_count > 0:
            idx = object_ids_ext_arr - 1
            data.dist_flat = full.dist_full[np.ix_(idx, idx)].copy()
        else:
            data.dist_flat = np.zeros((0, 0), dtype=np.int32)

        data.start_cost_flat = np.zeros((data.num_heroes, data.object_count), dtype=np.int32)

        for h in range(data.num_heroes):
            hs = hero_states_before_day[h]

            for j in range(data.object_count):
                obj_ext = data.object_ids_ext[j]

                if day == 1:
                    base_dist = int(full.dist_start_by_objid[obj_ext])
                    carry = 0
                else:
                    if hs.anchor_ext == 0:
                        base_dist = int(full.dist_start_by_objid[obj_ext])
                    else:
                        base_dist = full.dist_by_objid(hs.anchor_ext, obj_ext)
                    carry = hs.carry_discount

                if carry >= base_dist:
                    data.start_cost_flat[h, j] = VISIT_COST
                else:
                    data.start_cost_flat[h, j] = (base_dist - carry) + VISIT_COST

        return data


# ============================================================
# Решение одного дня
# ============================================================

class Solution:
    def __init__(self, data: DayData):
        self.data = data
        self.routes: List[List[int]] = [[] for _ in range(data.num_heroes)]
        self.route_costs: List[int] = [0 for _ in range(data.num_heroes)]
        self.obj_route: List[int] = [-1 for _ in range(data.object_count)]
        self.obj_pos: List[int] = [-1 for _ in range(data.object_count)]
        self.assigned_count: int = 0

    @staticmethod
    def empty(data: DayData) -> "Solution":
        return Solution(data)

    def clone(self) -> "Solution":
        other = Solution(self.data)
        other.routes = [r.copy() for r in self.routes]
        other.route_costs = self.route_costs.copy()
        other.obj_route = self.obj_route.copy()
        other.obj_pos = self.obj_pos.copy()
        other.assigned_count = self.assigned_count
        return other

    def assigned(self, obj: int) -> bool:
        return self.obj_route[obj] != -1

    def visited_count(self) -> int:
        return self.assigned_count

    def total_leftover(self) -> int:
        total = 0
        for r in range(self.data.num_heroes):
            total += max(0, self.data.hero_capacity(r) - self.route_costs[r])
        return total

    def total_used(self) -> int:
        total = 0
        for r in range(self.data.num_heroes):
            total += min(self.data.hero_capacity(r), self.route_costs[r])
        return total

    def quality_key(self) -> Tuple[int, int]:
        return self.visited_count(), self.total_leftover()

    def update_index_from(self, r: int, start_pos: int) -> None:
        if start_pos < 0:
            start_pos = 0
        route = self.routes[r]
        for pos in range(start_pos, len(route)):
            obj = route[pos]
            self.obj_route[obj] = r
            self.obj_pos[obj] = pos

    def removal_delta_by_pos(self, r: int, pos: int) -> int:
        route = self.routes[r]
        n = len(route)
        x = route[pos]

        if n == 1:
            return self.route_costs[r]

        if pos == 0:
            b = route[1]
            return self.data.start_cost(r, x) - self.data.start_cost(r, b) + self.data.dist(x, b) + VISIT_COST

        if pos == n - 1:
            a = route[n - 2]
            return self.data.dist(a, x) + VISIT_COST

        a = route[pos - 1]
        b = route[pos + 1]
        return self.data.dist(a, x) + self.data.dist(x, b) - self.data.dist(a, b) + VISIT_COST

    def removal_delta(self, obj: int) -> int:
        return self.removal_delta_by_pos(self.obj_route[obj], self.obj_pos[obj])

    def insertion_delta(self, r: int, obj: int, pos: int) -> int:
        route = self.routes[r]
        n = len(route)

        if n == 0:
            return self.data.start_cost(r, obj)

        if pos == 0:
            b = route[0]
            return self.data.start_cost(r, obj) - self.data.start_cost(r, b) + self.data.dist(obj, b) + VISIT_COST

        if pos == n:
            a = route[n - 1]
            return self.data.dist(a, obj) + VISIT_COST

        a = route[pos - 1]
        b = route[pos]
        return self.data.dist(a, obj) + self.data.dist(obj, b) - self.data.dist(a, b) + VISIT_COST

    def best_insertion_in_route(self, r: int, obj: int) -> Optional[Tuple[int, int]]:
        route = self.routes[r]
        n = len(route)

        cap_ext = self.data.hero_capacity(r) + VISIT_COST
        base = self.route_costs[r]

        found = False
        best_delta = 0
        best_pos = -1

        for pos in range(n + 1):
            delta = self.insertion_delta(r, obj, pos)
            if base + delta <= cap_ext:
                if (not found) or (delta < best_delta) or (delta == best_delta and pos < best_pos):
                    found = True
                    best_delta = delta
                    best_pos = pos

        if not found:
            return None

        return best_delta, best_pos

    def insert(self, obj: int, r: int, pos: int, given_delta: Optional[int] = None) -> None:
        if self.assigned(obj):
            raise RuntimeError("insert: объект уже назначен")

        delta = self.insertion_delta(r, obj, pos) if given_delta is None else given_delta
        self.routes[r].insert(pos, obj)
        self.route_costs[r] += delta
        self.obj_route[obj] = r
        self.assigned_count += 1
        self.update_index_from(r, pos)

    def remove_object(self, obj: int) -> int:
        if not self.assigned(obj):
            return 0

        r = self.obj_route[obj]
        pos = self.obj_pos[obj]
        delta = self.removal_delta_by_pos(r, pos)

        self.routes[r].pop(pos)
        self.route_costs[r] -= delta
        self.obj_route[obj] = -1
        self.obj_pos[obj] = -1
        self.assigned_count -= 1
        self.update_index_from(r, pos)
        return delta


# ============================================================
# LNS-операторы
# ============================================================

class DestroyOp:
    RANDOM = "random"
    WORST = "worst"


class RepairOp:
    GREEDY = "greedy"
    REGRET2 = "regret2"


def greedy_insert_one(sol: Solution) -> Optional[Dict]:
    best_obj = -1
    best_r = -1
    best_pos = -1
    best_delta = 0
    found = False

    for obj in range(sol.data.object_count):
        if sol.assigned(obj):
            continue

        for r in range(sol.data.num_heroes):
            ins = sol.best_insertion_in_route(r, obj)
            if ins is None:
                continue

                # unreachable

            delta, pos = ins

            if (
                (not found)
                or (delta < best_delta)
                or (delta == best_delta and r < best_r)
                or (delta == best_delta and r == best_r and pos < best_pos)
                or (delta == best_delta and r == best_r and pos == best_pos and obj < best_obj)
            ):
                found = True
                best_obj = obj
                best_r = r
                best_pos = pos
                best_delta = delta

    if not found:
        return None

    sol.insert(best_obj, best_r, best_pos, best_delta)
    return {"obj": best_obj, "route": best_r, "pos": best_pos, "delta": best_delta}


def repair_greedy(sol: Solution) -> None:
    while True:
        info = greedy_insert_one(sol)
        if info is None:
            break


def regret2_insert_one(sol: Solution) -> Optional[Dict]:
    chosen_obj = -1
    chosen_r = -1
    chosen_pos = -1
    chosen_best_delta = 0
    best_regret = -10**18
    found = False
    BIG_M = 1_000_000

    for obj in range(sol.data.object_count):
        if sol.assigned(obj):
            continue

        best1 = 10**18
        best2 = 10**18
        best_route = -1
        best_pos = -1

        for r in range(sol.data.num_heroes):
            ins = sol.best_insertion_in_route(r, obj)
            if ins is None:
                continue
            delta, pos = ins

            if delta < best1:
                best2 = best1
                best1 = delta
                best_route = r
                best_pos = pos
            elif delta < best2:
                best2 = delta

        if best_route == -1:
            continue

        if best2 >= 10**18:
            best2 = best1 + BIG_M

        regret = best2 - best1

        if (
            (not found)
            or (regret > best_regret)
            or (regret == best_regret and best1 < chosen_best_delta)
            or (regret == best_regret and best1 == chosen_best_delta and obj < chosen_obj)
        ):
            found = True
            best_regret = regret
            chosen_obj = obj
            chosen_r = best_route
            chosen_pos = best_pos
            chosen_best_delta = best1

    if not found:
        return None

    sol.insert(chosen_obj, chosen_r, chosen_pos, chosen_best_delta)
    return {
        "obj": chosen_obj,
        "route": chosen_r,
        "pos": chosen_pos,
        "delta": chosen_best_delta,
        "regret": best_regret,
    }


# ============================================================
# Выбор объектов и warmup
# ============================================================

def select_objects_for_day(
    full: FullData,
    day: int,
    limit: int,
    mode: str,
    seed: int,
) -> List[int]:
    all_ids = (np.where(full.object_day_open[1:] == day)[0] + 1).astype(np.int32).tolist()

    if limit <= 0 or limit >= len(all_ids):
        return sorted(all_ids)

    rng = random.Random(seed)

    if mode == "nearest":
        all_ids.sort(key=lambda obj: (int(full.dist_start_by_objid[obj]), obj))
        return sorted(all_ids[:limit])

    if mode == "random":
        return sorted(rng.sample(all_ids, k=limit))

    all_ids_sorted = sorted(all_ids, key=lambda obj: (int(full.dist_start_by_objid[obj]), obj))
    near_k = limit // 2
    near_part = all_ids_sorted[:near_k]
    near_set = set(near_part)
    rest_pool = [x for x in all_ids if x not in near_set]
    rand_k = limit - len(near_part)
    rand_part = rng.sample(rest_pool, k=rand_k)
    return sorted(near_part + rand_part)


def update_hero_states_from_solution(
    hero_states: List[HeroState],
    day_data: DayData,
    sol: Solution,
) -> None:
    for h in range(day_data.num_heroes):
        cap = day_data.hero_capacity(h)

        if len(sol.routes[h]) > 0:
            last_obj_internal = sol.routes[h][-1]
            hero_states[h].anchor_ext = day_data.object_id(last_obj_internal)
            hero_states[h].carry_discount = max(0, cap - sol.route_costs[h])
        else:
            hero_states[h].carry_discount += cap


def build_hero_states_before_day(full: FullData, cfg: AppConfig) -> List[HeroState]:
    hero_states = [HeroState(anchor_ext=0, carry_discount=0) for _ in range(cfg.heroes)]

    if not cfg.warmup_previous or cfg.day <= 1:
        return hero_states

    log_console("Прогрев предыдущих дней включён")

    for d in range(1, cfg.day):
        chosen = select_objects_for_day(
            full=full,
            day=d,
            limit=cfg.warmup_max_objects,
            mode="nearest",
            seed=cfg.seed + 1000 * d,
        ) if cfg.warmup_max_objects > 0 else None

        day_data = DayData.build_for_day(
            full=full,
            day=d,
            heroes_count=cfg.heroes,
            hero_states_before_day=hero_states,
            selected_object_ids_ext=chosen,
        )

        sol = Solution.empty(day_data)
        repair_greedy(sol)
        update_hero_states_from_solution(hero_states, day_data, sol)

        log_console(
            f"Прогрев day {d}: objects={day_data.object_count}, "
            f"visited={sol.visited_count()}, leftover={sol.total_leftover()}"
        )

    return hero_states


# ============================================================
# 2D layout
# ============================================================

def classical_mds(dist_matrix: np.ndarray, dim: int = 2) -> np.ndarray:
    n = dist_matrix.shape[0]
    if n == 0:
        return np.zeros((0, dim), dtype=np.float64)
    if n == 1:
        return np.zeros((1, dim), dtype=np.float64)

    d2 = dist_matrix.astype(np.float64) ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j

    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    use_vals = np.maximum(eigvals[:dim], 0.0)
    use_vecs = eigvecs[:, :dim]
    coords = use_vecs * np.sqrt(use_vals)

    if coords.shape[1] < dim:
        coords = np.hstack([coords, np.zeros((coords.shape[0], dim - coords.shape[1]))])

    return coords


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords.copy()

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) * 0.5
    scale = float(np.max(maxs - mins))
    if scale < 1e-9:
        scale = 1.0

    return (coords - center) / scale + 0.5


def push_points_away_from_depot(coords: np.ndarray, depot_idx: int = 0) -> np.ndarray:
    n = coords.shape[0]
    if n <= 1:
        return coords.copy()

    out = coords.astype(np.float64, copy=True)
    depot = out[depot_idx].copy()

    # Небольшое глобальное растяжение от центра таверны
    for i in range(n):
        if i == depot_idx:
            continue
        v = out[i] - depot
        d = float(np.hypot(v[0], v[1]))
        if d < 1e-9:
            angle = (i * 2.399963229728653) % (2 * math.pi)
            v = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            d = 1.0

        scale = 1.12
        out[i] = depot + v * scale

    # Жёсткая "зона свободного места" вокруг таверны
    min_r = min(0.19, max(0.12, 1.45 / math.sqrt(max(4, n))))
    for i in range(n):
        if i == depot_idx:
            continue
        v = out[i] - depot
        d = float(np.hypot(v[0], v[1]))
        if d < 1e-9:
            angle = (i * 2.399963229728653) % (2 * math.pi)
            v = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            d = 1.0
        if d < min_r:
            out[i] = depot + v / d * min_r

    out[:, 0] = np.clip(out[:, 0], 0.02, 0.98)
    out[:, 1] = np.clip(out[:, 1], 0.02, 0.98)
    out[depot_idx] = depot
    return out


def spread_out_coords(coords: np.ndarray, seed: int, keep_first_fixed: bool = True) -> np.ndarray:
    n = coords.shape[0]
    if n <= 1:
        return coords.copy()

    out = coords.astype(np.float64, copy=True)
    rng = np.random.RandomState(seed)
    out += rng.normal(scale=0.0025, size=out.shape)
    original = out.copy()

    if n <= 20:
        iters = 75
    elif n <= 60:
        iters = 48
    else:
        iters = 28

    min_sep = min(0.16, max(0.050, 0.58 / math.sqrt(max(2, n))))
    eps = 1e-9

    for _ in range(iters):
        diff = out[:, None, :] - out[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + eps)
        mask = (dist < min_sep) & (dist > 0.0)
        if not np.any(mask):
            break

        force = np.where(mask, (min_sep - dist) / (dist + eps) * 0.020, 0.0)
        disp = np.sum(diff * force[:, :, None], axis=1)

        out += disp
        out += (original - out) * 0.045

        if keep_first_fixed:
            out[0] = original[0]

        out[:, 0] = np.clip(out[:, 0], 0.02, 0.98)
        out[:, 1] = np.clip(out[:, 1], 0.02, 0.98)

    return out


@dataclass
class LayoutData:
    coord_by_ext: Dict[int, Tuple[float, float]]
    selected_set: set
    anchor_heroes_by_ext: Dict[int, List[int]]

    def point(self, ext: int, rect: pygame.Rect, pad: int = 18) -> Tuple[int, int]:
        if ext not in self.coord_by_ext:
            return rect.center

        x_norm, y_norm = self.coord_by_ext[ext]
        inner = rect.inflate(-2 * pad, -2 * pad)
        x = inner.x + x_norm * inner.w
        y = inner.y + (1.0 - y_norm) * inner.h
        return int(x), int(y)


def build_layout_data(
    full: FullData,
    day_data: DayData,
    hero_states_before_day: List[HeroState],
    seed: int,
) -> LayoutData:
    selected_ext = list(day_data.object_ids_ext)
    selected_set = set(selected_ext)

    anchor_heroes_by_ext: Dict[int, List[int]] = defaultdict(list)
    extra_anchor_ext = []

    for h, hs in enumerate(hero_states_before_day, start=1):
        if hs.anchor_ext != 0:
            anchor_heroes_by_ext[hs.anchor_ext].append(h)
            if hs.anchor_ext not in selected_set:
                extra_anchor_ext.append(hs.anchor_ext)

    extra_anchor_ext = sorted(set(extra_anchor_ext))
    point_exts = [0] + selected_ext + extra_anchor_ext

    n = len(point_exts)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = ext_distance_from_full(full, point_exts[i], point_exts[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    coords = classical_mds(dist_matrix, dim=2)
    coords = normalize_coords(coords)
    coords = spread_out_coords(coords, seed=seed, keep_first_fixed=True)
    coords = push_points_away_from_depot(coords, depot_idx=0)

    coord_by_ext = {ext: (float(coords[i, 0]), float(coords[i, 1])) for i, ext in enumerate(point_exts)}

    return LayoutData(
        coord_by_ext=coord_by_ext,
        selected_set=selected_set,
        anchor_heroes_by_ext=dict(anchor_heroes_by_ext),
    )


# ============================================================
# Лог, hover, scroll, help layout
# ============================================================

@dataclass
class LogEntry:
    iter_no: int
    substep: int
    stage: str
    text: str
    highlight_exts: List[int] = field(default_factory=list)
    color: Tuple[int, int, int] = TEXT


@dataclass
class HoveredObjectInfo:
    panel_key: str
    ext: int
    pos: Tuple[int, int]
    dist_px: float


@dataclass
class HoveredSegmentInfo:
    panel_key: str
    panel_title: str
    hero_idx: int
    from_ext: int
    to_ext: int
    length: int
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    dist_px: float


@dataclass
class ScrollView:
    area_rect: pygame.Rect
    track_rect: Optional[pygame.Rect]
    viewport_h: int
    content_h: int
    offset_px: int
    max_offset_px: int
    scrollbar_w: int = SCROLLBAR_W
    scrollbar_gap: int = SCROLLBAR_GAP
    min_thumb_h: int = SCROLLBAR_MIN_THUMB_H


@dataclass
class ScrollDragState:
    target: Optional[str] = None
    grab_offset_y: int = 0


@dataclass
class HelpRenderItem:
    text: Optional[str]
    font: Optional[Any]
    color: Tuple[int, int, int]
    height: int


@dataclass
class HelpLayout:
    panel_rect: pygame.Rect
    title_font: Any
    title_lines: List[str]
    hint_font: Any
    hint_text: str
    content_view: ScrollView
    items: List[HelpRenderItem]


# ============================================================
# Снимок состояния stepper для undo
# ============================================================

@dataclass
class StepperSnapshot:
    iteration: int
    substep_in_iter: int
    init_substep: int
    global_step: int
    accepted_moves: int
    improving_moves: int
    best_updates: int
    current: Solution
    best: Solution
    candidate: Solution
    stage: str
    temperature: float
    destroy_op: str
    repair_op: str
    q: int
    q_done: int
    random_destroy_order: List[int]
    highlight_removed_ext: set
    highlight_inserted_ext: Optional[int]
    last_accept_result: Optional[bool]
    log_entries: deque
    rng_state: object


# ============================================================
# LNS stepper
# ============================================================

class LNSStepper:
    def __init__(
        self,
        day_data: DayData,
        hero_states_before_day: List[HeroState],
        cfg: AppConfig,
        seed: int,
    ):
        self.day_data = day_data
        self.hero_states_before_day = hero_states_before_day
        self.cfg = cfg
        self.seed = seed
        self.rng = random.Random(seed)

        self.iteration = 0
        self.substep_in_iter = 0
        self.init_substep = 0
        self.global_step = 0

        self.accepted_moves = 0
        self.improving_moves = 0
        self.best_updates = 0

        self.current = Solution.empty(day_data)
        self.best = Solution.empty(day_data)
        self.candidate = Solution.empty(day_data)

        self.stage = "init"
        self.temperature = cfg.temp_start

        self.destroy_op = DestroyOp.RANDOM
        self.repair_op = RepairOp.GREEDY
        self.q = 0
        self.q_done = 0

        self.random_destroy_order: List[int] = []

        self.highlight_removed_ext: set = set()
        self.highlight_inserted_ext: Optional[int] = None
        self.last_accept_result: Optional[bool] = None

        self.log_entries = deque(maxlen=22)
        self._push_log("init", "Начинаем построение стартового решения", [])

    def make_snapshot(self) -> StepperSnapshot:
        return StepperSnapshot(
            iteration=self.iteration,
            substep_in_iter=self.substep_in_iter,
            init_substep=self.init_substep,
            global_step=self.global_step,
            accepted_moves=self.accepted_moves,
            improving_moves=self.improving_moves,
            best_updates=self.best_updates,
            current=self.current.clone(),
            best=self.best.clone(),
            candidate=self.candidate.clone(),
            stage=self.stage,
            temperature=self.temperature,
            destroy_op=self.destroy_op,
            repair_op=self.repair_op,
            q=self.q,
            q_done=self.q_done,
            random_destroy_order=self.random_destroy_order.copy(),
            highlight_removed_ext=set(self.highlight_removed_ext),
            highlight_inserted_ext=self.highlight_inserted_ext,
            last_accept_result=self.last_accept_result,
            log_entries=deque(self.log_entries, maxlen=self.log_entries.maxlen),
            rng_state=self.rng.getstate(),
        )

    def restore_snapshot(self, snap: StepperSnapshot) -> None:
        self.iteration = snap.iteration
        self.substep_in_iter = snap.substep_in_iter
        self.init_substep = snap.init_substep
        self.global_step = snap.global_step
        self.accepted_moves = snap.accepted_moves
        self.improving_moves = snap.improving_moves
        self.best_updates = snap.best_updates

        self.current = snap.current.clone()
        self.best = snap.best.clone()
        self.candidate = snap.candidate.clone()

        self.stage = snap.stage
        self.temperature = snap.temperature
        self.destroy_op = snap.destroy_op
        self.repair_op = snap.repair_op
        self.q = snap.q
        self.q_done = snap.q_done

        self.random_destroy_order = snap.random_destroy_order.copy()
        self.highlight_removed_ext = set(snap.highlight_removed_ext)
        self.highlight_inserted_ext = snap.highlight_inserted_ext
        self.last_accept_result = snap.last_accept_result

        self.log_entries = deque(snap.log_entries, maxlen=snap.log_entries.maxlen)
        self.rng.setstate(snap.rng_state)

    def _stage_color(self, stage: str, text: str) -> Tuple[int, int, int]:
        if stage == "destroy":
            return REMOVED
        if stage == "repair":
            return INSERTED
        if stage == "accept":
            return REJECT if "отклонён" in text else ACCEPT
        if stage == "init":
            return (150, 190, 255)
        return TEXT

    def _push_log(self, stage: str, text: str, highlight_exts: List[int]) -> None:
        if self.stage == "init" or stage == "init":
            iter_no = 0
            substep = self.init_substep
        else:
            iter_no = self.iteration
            substep = self.substep_in_iter

        self.log_entries.appendleft(
            LogEntry(
                iter_no=iter_no,
                substep=substep,
                stage=stage,
                text=text,
                highlight_exts=list(highlight_exts),
                color=self._stage_color(stage, text),
            )
        )

    def compute_temperature(self, progress: float) -> float:
        if self.cfg.temp_start <= 0.0 or self.cfg.temp_end <= 0.0:
            return 0.0

        progress = clamp(progress, 0.0, 1.0)
        if abs(self.cfg.temp_start - self.cfg.temp_end) < 1e-15:
            return self.cfg.temp_start

        return self.cfg.temp_start * ((self.cfg.temp_end / self.cfg.temp_start) ** progress)

    def choose_q(self, sol: Solution) -> int:
        if sol.visited_count() <= 0:
            return 0

        lo = max(1, int(math.floor(self.cfg.destroy_frac_min * sol.visited_count())))
        hi = max(lo, int(math.ceil(self.cfg.destroy_frac_max * sol.visited_count())))
        hi = min(hi, sol.visited_count())
        lo = min(lo, hi)

        return self.rng.randint(lo, hi)

    def accept(self, cand: Solution, cur: Solution, temperature: float) -> bool:
        ck = cand.quality_key()
        uk = cur.quality_key()

        if ck == uk or better_key(ck, uk):
            return True

        delta = (
            (cand.visited_count() - cur.visited_count())
            + (cand.total_leftover() - cur.total_leftover()) / 1_000_000.0
        )

        if temperature <= 0.0:
            return False

        prob = math.exp(delta / temperature)
        return self.rng.random() < prob

    def _random_destroy_one(self) -> bool:
        if self.q_done >= self.q or not self.random_destroy_order:
            return False

        obj = self.random_destroy_order.pop()
        ext = self.day_data.object_id(obj)
        delta = self.candidate.remove_object(obj)

        self.highlight_removed_ext.add(ext)
        if self.highlight_inserted_ext == ext:
            self.highlight_inserted_ext = None

        self.q_done += 1
        self.substep_in_iter += 1
        self._push_log(
            "destroy",
            f"destroy random: удалили мельницу {ext}, delta={delta}, осталось удалить {self.q - self.q_done}",
            [ext],
        )
        return True

    def _worst_destroy_one(self) -> bool:
        if self.q_done >= self.q or self.candidate.visited_count() == 0:
            return False

        cands = []
        for obj in range(self.day_data.object_count):
            if self.candidate.assigned(obj):
                cands.append((self.candidate.removal_delta(obj), obj))

        cands.sort(key=lambda x: (-x[0], x[1]))
        limit = min(len(cands), max(1, self.cfg.rcl_size))
        idx = self.rng.randint(0, limit - 1)

        delta, obj = cands[idx]
        ext = self.day_data.object_id(obj)
        self.candidate.remove_object(obj)

        self.highlight_removed_ext.add(ext)
        if self.highlight_inserted_ext == ext:
            self.highlight_inserted_ext = None

        self.q_done += 1
        self.substep_in_iter += 1
        self._push_log(
            "destroy",
            f"destroy worst: удалили мельницу {ext}, delta={delta}, осталось удалить {self.q - self.q_done}",
            [ext],
        )
        return True

    def _repair_one(self) -> bool:
        if self.repair_op == RepairOp.GREEDY:
            info = greedy_insert_one(self.candidate)
            if info is None:
                return False

            ext = self.day_data.object_id(info["obj"])
            self.highlight_inserted_ext = ext
            self.highlight_removed_ext.discard(ext)

            self.substep_in_iter += 1
            self._push_log(
                "repair",
                f"repair greedy: вставили мельницу {ext} у героя {info['route'] + 1}, pos={info['pos']}, delta={info['delta']}",
                [ext],
            )
            return True

        info = regret2_insert_one(self.candidate)
        if info is None:
            return False

        ext = self.day_data.object_id(info["obj"])
        self.highlight_inserted_ext = ext
        self.highlight_removed_ext.discard(ext)

        self.substep_in_iter += 1
        self._push_log(
            "repair",
            f"repair regret2: вставили мельницу {ext} у героя {info['route'] + 1}, pos={info['pos']}, delta={info['delta']}, regret={info['regret']}",
            [ext],
        )
        return True

    def micro_step(self) -> None:
        if self.stage == "done":
            return

        self.global_step += 1

        if self.stage == "init":
            info = greedy_insert_one(self.current)

            if info is not None:
                self.best = self.current.clone()
                self.candidate = self.current.clone()

                ext = self.day_data.object_id(info["obj"])
                self.highlight_inserted_ext = ext
                self.init_substep += 1
                self._push_log(
                    "init",
                    f"init: вставили мельницу {ext} у героя {info['route'] + 1}, pos={info['pos']}, delta={info['delta']}",
                    [ext],
                )
                return

            self.best = self.current.clone()
            self.candidate = self.current.clone()
            self.highlight_inserted_ext = None
            self.stage = "iter_start"
            self.init_substep += 1
            self._push_log(
                "init",
                f"Стартовое решение готово: visited={self.current.visited_count()}, leftover={self.current.total_leftover()}",
                [],
            )
            return

        if self.stage == "iter_start":
            self.last_accept_result = None
            self.highlight_removed_ext.clear()
            self.highlight_inserted_ext = None

            if self.cfg.iterations > 0 and self.iteration >= self.cfg.iterations:
                self.stage = "done"
                self._push_log(
                    "done",
                    f"Лимит итераций достигнут. Лучшее решение: visited={self.best.visited_count()}, leftover={self.best.total_leftover()}",
                    [],
                )
                return

            self.iteration += 1
            self.substep_in_iter = 0

            progress = self.iteration / max(1, self.cfg.iterations) if self.cfg.iterations > 0 else min(1.0, self.iteration / 300.0)
            self.temperature = self.compute_temperature(progress)

            self.destroy_op = DestroyOp.RANDOM if self.rng.random() < 0.5 else DestroyOp.WORST
            self.repair_op = RepairOp.GREEDY if self.rng.random() < 0.5 else RepairOp.REGRET2

            self.candidate = self.current.clone()
            self.q = self.choose_q(self.candidate)
            self.q_done = 0

            if self.destroy_op == DestroyOp.RANDOM:
                assigned_objs = [obj for obj in range(self.day_data.object_count) if self.candidate.assigned(obj)]
                self.rng.shuffle(assigned_objs)
                self.random_destroy_order = assigned_objs[:self.q]
            else:
                self.random_destroy_order = []

            self.stage = "destroy" if self.q > 0 else "repair"
            self._push_log(
                "iter_start",
                f"Итерация {self.iteration}: destroy={self.destroy_op}, repair={self.repair_op}, q={self.q}, temp={self.temperature:.3f}",
                [],
            )
            return

        if self.stage == "destroy":
            changed = self._random_destroy_one() if self.destroy_op == DestroyOp.RANDOM else self._worst_destroy_one()
            if changed:
                return

            self._push_log("destroy", "Destroy-фаза завершена, начинаем repair", [])
            self.stage = "repair"
            return

        if self.stage == "repair":
            changed = self._repair_one()
            if changed:
                return

            accept_it = self.accept(self.candidate, self.current, self.temperature)
            cur_key = self.current.quality_key()
            cand_key = self.candidate.quality_key()

            self.substep_in_iter += 1

            if accept_it:
                if better_key(cand_key, cur_key):
                    self.improving_moves += 1

                self.current = self.candidate.clone()
                self.accepted_moves += 1
                self.last_accept_result = True

                if better_key(self.current.quality_key(), self.best.quality_key()):
                    self.best = self.current.clone()
                    self.best_updates += 1
                    self._push_log(
                        "accept",
                        f"accept: кандидат принят, найден новый BEST (visited={self.best.visited_count()}, leftover={self.best.total_leftover()})",
                        [],
                    )
                else:
                    self._push_log("accept", "accept: кандидат принят", [])
            else:
                self.last_accept_result = False
                self._push_log("accept", "accept: кандидат отклонён", [])

            self.stage = "post_accept"
            return

        if self.stage == "post_accept":
            self.stage = "iter_start"

    def stage_label(self) -> str:
        return self.stage


# ============================================================
# Шрифты и базовый текстовый рендер
# ============================================================

class FontManager:
    def __init__(self, render_scale: float = 1.0):
        self.cache = {}
        self.render_scale = float(render_scale)
        self.candidates = [
            "segoeui",
            "dejavusans",
            "notosans",
            "liberationsans",
            "arial",
            "verdana",
            "tahoma",
        ]
        self.default_font_name = pygame.font.get_default_font()

    def _find_path(self, bold: bool) -> Optional[str]:
        for name in self.candidates:
            path = pygame.font.match_font(name, bold=bold)
            if path:
                return path
        try:
            return self.default_font_name
        except Exception:
            return None

    def get(self, size: int, bold: bool = False):
        scaled = max(10, int(round(size * UI_SCALE * self.render_scale)))
        key = (scaled, bold)
        if key not in self.cache:
            path = self._find_path(bold=bold)
            if path is not None:
                font = pygame.font.Font(path, scaled)
            else:
                font = pygame.font.SysFont(None, scaled, bold=bold)
            self.cache[key] = font
        return self.cache[key]


def ellipsize(font, text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if font.size(text)[0] <= max_width:
        return text

    suffix = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = text[:mid] + suffix
        if font.size(cand)[0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + suffix


def draw_text(
    surf: pygame.Surface,
    text: str,
    pos: Tuple[int, int],
    font,
    color,
    max_width: Optional[int] = None,
) -> pygame.Rect:
    if max_width is not None:
        text = ellipsize(font, text, max_width)
    img = font.render(text, True, color)
    rect = img.get_rect(topleft=pos)
    surf.blit(img, rect)
    return rect


def draw_centered_text(
    surf: pygame.Surface,
    text: str,
    rect: pygame.Rect,
    font,
    color,
    max_width: Optional[int] = None,
) -> pygame.Rect:
    if max_width is None:
        max_width = rect.w - 6
    text = ellipsize(font, text, max_width)
    img = font.render(text, True, color)
    img_rect = img.get_rect(center=rect.center)
    surf.blit(img, img_rect)
    return img_rect


def draw_outlined_text(
    surf: pygame.Surface,
    text: str,
    pos: Tuple[int, int],
    font,
    color,
    outline_color=(10, 10, 10),
) -> pygame.Rect:
    base = font.render(text, True, color)
    outline = font.render(text, True, outline_color)
    rect = base.get_rect(topleft=pos)

    x, y = pos
    for ox, oy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        surf.blit(outline, (x + ox, y + oy))
    surf.blit(base, rect)
    return rect


def wrap_text(font, text: str, max_width: int) -> List[str]:
    if max_width <= 0:
        return [""]
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = ""

    def push_long_word(word: str) -> None:
        nonlocal cur
        part = ""
        for ch in word:
            test = part + ch
            if not part or font.size(test)[0] <= max_width:
                part = test
            else:
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(part)
                part = ch
        if part:
            if not cur:
                cur = part
            else:
                cand = cur + " " + part
                if font.size(cand)[0] <= max_width:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = part

    for word in words:
        if not cur:
            if font.size(word)[0] <= max_width:
                cur = word
            else:
                push_long_word(word)
            continue

        cand = cur + " " + word
        if font.size(cand)[0] <= max_width:
            cur = cand
        else:
            lines.append(cur)
            cur = ""
            if font.size(word)[0] <= max_width:
                cur = word
            else:
                push_long_word(word)

    if cur:
        lines.append(cur)

    return lines if lines else [""]


def draw_diamond(surf, color, center, r, width=0):
    x, y = center
    pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
    pygame.draw.polygon(surf, color, pts, width)


# ============================================================
# Универсальные UI компоненты
# ============================================================

@dataclass
class UIButton:
    name: str
    label: str
    rect: pygame.Rect
    toggled: bool = False
    enabled: bool = True


@dataclass
class UISpeedBlock:
    group_rect: pygame.Rect
    minus_rect: pygame.Rect
    plus_rect: pygame.Rect
    value_rect: pygame.Rect


def make_scroll_view(
    area_rect: pygame.Rect,
    content_h: int,
    offset_px: int,
    scrollbar_w: int = SCROLLBAR_W,
    scrollbar_gap: int = SCROLLBAR_GAP,
    min_thumb_h: int = SCROLLBAR_MIN_THUMB_H,
) -> ScrollView:
    base = area_rect.copy()

    if content_h <= base.h:
        return ScrollView(
            area_rect=base,
            track_rect=None,
            viewport_h=base.h,
            content_h=content_h,
            offset_px=0,
            max_offset_px=0,
            scrollbar_w=scrollbar_w,
            scrollbar_gap=scrollbar_gap,
            min_thumb_h=min_thumb_h,
        )

    content_rect = base.copy()
    content_rect.w -= (scrollbar_w + scrollbar_gap)
    content_rect.w = max(10, content_rect.w)

    max_offset = max(0, content_h - content_rect.h)
    offset = int(clamp(offset_px, 0, max_offset))
    track_rect = pygame.Rect(content_rect.right + scrollbar_gap, base.y, scrollbar_w, base.h)

    return ScrollView(
        area_rect=content_rect,
        track_rect=track_rect,
        viewport_h=content_rect.h,
        content_h=content_h,
        offset_px=offset,
        max_offset_px=max_offset,
        scrollbar_w=scrollbar_w,
        scrollbar_gap=scrollbar_gap,
        min_thumb_h=min_thumb_h,
    )


def scrollbar_thumb_rect(view: ScrollView) -> Optional[pygame.Rect]:
    if view.track_rect is None or view.max_offset_px <= 0 or view.content_h <= 0:
        return None

    thumb_h = max(
        view.min_thumb_h,
        int(view.track_rect.h * view.viewport_h / max(1, view.content_h))
    )
    thumb_h = min(thumb_h, view.track_rect.h)

    travel = max(1, view.track_rect.h - thumb_h)
    thumb_y = view.track_rect.y + int(travel * view.offset_px / max(1, view.max_offset_px))
    return pygame.Rect(view.track_rect.x + 1, thumb_y, max(2, view.track_rect.w - 2), thumb_h)


def draw_scrollbar(surface: pygame.Surface, view: ScrollView, mouse_pos: Tuple[int, int]) -> None:
    if view.track_rect is None:
        return

    thumb = scrollbar_thumb_rect(view)
    if thumb is None:
        return

    hovered_track = view.track_rect.collidepoint(mouse_pos)
    hovered_thumb = thumb.collidepoint(mouse_pos)

    track_radius = max(4, view.track_rect.w // 2)
    thumb_radius = max(4, thumb.w // 2)

    pygame.draw.rect(surface, SCROLL_TRACK, view.track_rect, border_radius=track_radius)
    pygame.draw.rect(surface, BORDER, view.track_rect, width=1, border_radius=track_radius)
    pygame.draw.rect(
        surface,
        SCROLL_THUMB_HOVER if (hovered_track or hovered_thumb) else SCROLL_THUMB,
        thumb,
        border_radius=thumb_radius,
    )


def draw_button(
    surface: pygame.Surface,
    btn: UIButton,
    font,
    mouse_pos: Tuple[int, int],
    centered: bool = False,
) -> None:
    hovered = btn.rect.collidepoint(mouse_pos)

    fill = BUTTON_FILL if btn.enabled else BUTTON_DISABLED
    text_color = TEXT if btn.enabled else BUTTON_DISABLED_TEXT

    if btn.toggled and btn.enabled:
        fill = BUTTON_ACTIVE
    elif hovered and btn.enabled:
        fill = BUTTON_HOVER

    pygame.draw.rect(surface, fill, btn.rect, border_radius=7)
    pygame.draw.rect(surface, BUTTON_BORDER, btn.rect, width=1, border_radius=7)

    if centered:
        draw_centered_text(surface, btn.label, btn.rect, font, text_color, max_width=btn.rect.w - 12)
    else:
        draw_text(
            surface,
            btn.label,
            (btn.rect.x + 10, btn.rect.y + btn.rect.h // 2 - font.get_height() // 2),
            font,
            text_color,
            max_width=btn.rect.w - 18,
        )


# ============================================================
# Форматирование маршрутов
# ============================================================

def format_route_line(sol: Solution, hero_idx: int, hero_states_before_day: List[HeroState]) -> str:
    cap = sol.data.hero_capacity(hero_idx)
    used = sol.route_costs[hero_idx]
    start_ext = hero_states_before_day[hero_idx].anchor_ext
    start_txt = compact_ext_name(start_ext)

    route = sol.routes[hero_idx]
    if not route:
        return f"Г{hero_idx + 1:02d} {used}/{cap} | {start_txt}"

    objs = [str(sol.data.object_id(obj)) for obj in route]
    chain = ROUTE_ARROW.join([start_txt] + objs)
    return f"Г{hero_idx + 1:02d} {used}/{cap} | {chain}"


# ============================================================
# Состояние приложения
# ============================================================

@dataclass
class AppState:
    full: FullData
    cfg: AppConfig
    hero_states_before_day: List[HeroState]
    day_data: DayData
    layout: LayoutData
    stepper: LNSStepper
    chosen_objects_ext: List[int]


def build_app_state(full: FullData, cfg: AppConfig, seed: int) -> AppState:
    t0 = now_sec()

    hero_states_before_day = build_hero_states_before_day(full, cfg)

    chosen_objects_ext = select_objects_for_day(
        full=full,
        day=cfg.day,
        limit=cfg.objects,
        mode=cfg.sample_mode,
        seed=seed,
    )

    day_data = DayData.build_for_day(
        full=full,
        day=cfg.day,
        heroes_count=cfg.heroes,
        hero_states_before_day=hero_states_before_day,
        selected_object_ids_ext=chosen_objects_ext,
    )

    if day_data.object_count == 0:
        raise RuntimeError(f"Для дня {cfg.day} нет объектов")

    layout = build_layout_data(full, day_data, hero_states_before_day, seed=seed)
    stepper = LNSStepper(day_data, hero_states_before_day, cfg, seed)

    log_console(
        f"Подзадача готова: day={cfg.day}, heroes={cfg.heroes}, "
        f"objects={day_data.object_count}, sample={cfg.sample_mode}, seed={seed}, "
        f"time={now_sec() - t0:.2f} сек"
    )

    return AppState(
        full=full,
        cfg=cfg,
        hero_states_before_day=hero_states_before_day,
        day_data=day_data,
        layout=layout,
        stepper=stepper,
        chosen_objects_ext=chosen_objects_ext,
    )


# ============================================================
# Главное приложение
# ============================================================

class VisualizerApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        pygame.init()
        pygame.font.init()

        self.display_surface = pygame.display.set_mode(
            (cfg.width, cfg.height),
            pygame.RESIZABLE | pygame.DOUBLEBUF,
        )
        self.screen = self.display_surface
        self.present_surface: Optional[pygame.Surface] = None

        self.is_fullscreen = False
        self.windowed_size = (cfg.width, cfg.height)

        self.clock = pygame.time.Clock()
        self.fonts = FontManager(render_scale=cfg.supersampling)

        self.running = True
        self.autoplay = True
        self.show_ids = False
        self.show_help = False

        self.steps_per_sec = int(cfg.steps_per_sec)
        self.step_accum = 0.0
        self.seed = cfg.seed
        self.texture_seed = int(cfg.texture_seed)

        self.resize_loading = False
        self.resize_ready_at = 0.0
        self.resize_debounce_sec = 0.20
        self.resize_pending_size: Optional[Tuple[int, int]] = None

        self.texture_cache: Dict[Tuple[str, int, int, int], pygame.Surface] = {}
        self.texture_cache_limit = 72

        self.full: Optional[FullData] = None
        self.state: Optional[AppState] = None

        self.buttons: List[UIButton] = []
        self.speed_block: Optional[UISpeedBlock] = None

        self.log_scroll_px = 0
        self.routes_scroll_px = 0
        self.help_scroll_px = 0

        self.scroll_drag = ScrollDragState()

        self.routes_solution_mode = "candidate"
        self.routes_solution_order = ["candidate", "best", "current"]

        self.undo_history: deque = deque(maxlen=UNDO_HISTORY_LIMIT)

        self.recreate_render_target()

        pygame.display.set_caption("Визуализатор LNS | загрузка...")
        self.draw_loading_screen("Загрузка данных...", "Окно уже открыто. Подготавливаем визуализатор.")
        pygame.event.pump()

        self.full = FullData.load(cfg.data_dir)

        self.draw_loading_screen("Подготовка визуализатора...", "Строим подзадачу и раскладку карты.")
        pygame.event.pump()

        self.state = build_app_state(self.full, self.cfg, self.seed)
        self.relayout_ui()
        self.update_window_title()

    def prune_texture_cache(self) -> None:
        while len(self.texture_cache) > self.texture_cache_limit:
            self.texture_cache.popitem(last=False)  # удалить самый старый
        
    def panel_parchment_rect(self, rect: pygame.Rect) -> pygame.Rect:
        bronze_rect = rect.inflate(-self.px(12), -self.px(12))
        parchment_rect = bronze_rect.inflate(-self.px(4), -self.px(4))
        return parchment_rect

    def begin_resize_loading(self, w: int, h: int) -> None:
        w = max(980, int(w))
        h = max(650, int(h))

        self.resize_pending_size = (w, h)
        self.resize_ready_at = now_sec() + self.resize_debounce_sec
        self.resize_loading = True

        self.windowed_size = (w, h)
        self.display_surface = pygame.display.set_mode((w, h), pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.recreate_render_target()
        self.relayout_ui()

    def preload_visible_textures(self) -> None:
        rects = self.compute_layout()

        jobs = []

        # parchment-панели
        for key in ["current", "candidate", "best", "info", "log", "routes"]:
            pr = self.panel_parchment_rect(rects[key])
            jobs.append(("parchment", pr.w, pr.h))

        # карты
        for key in ["current", "candidate", "best"]:
            mr = self.map_inner_rect(rects[key])
            jobs.append(("map", mr.w, mr.h))

        # убираем дубликаты, но сохраняем порядок
        uniq = []
        seen = set()
        for job in jobs:
            if job not in seen:
                seen.add(job)
                uniq.append(job)

        total = len(uniq)
        for i, (kind, w, h) in enumerate(uniq, start=1):
            self.draw_loading_screen(
                "Генерация текстур...",
                f"Подготавливаем фон интерфейса: {i}/{total} ({kind}, {w}x{h})"
            )
            pygame.event.pump()
            self.get_textured_surface(kind, (w, h))

    def finalize_resize_if_needed(self) -> None:
        if not self.resize_loading:
            return
        if self.resize_pending_size is None:
            return
        if now_sec() < self.resize_ready_at:
            return

        self.draw_loading_screen(
            "Генерация текстур...",
            "Подождите, пересобираем текстуры после изменения размера окна."
        )
        pygame.event.pump()

        self.preload_visible_textures()
        self.resize_loading = False
        self.resize_pending_size = None

    # --------------------------------------------------------
    # Supersampling helpers
    # --------------------------------------------------------
    def px(self, x: float, min_value: int = 1) -> int:
        return max(min_value, int(round(x * self.cfg.supersampling)))

    def px0(self, x: float) -> int:
        return int(round(x * self.cfg.supersampling))

    def to_render_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        s = self.cfg.supersampling
        if s <= 1.0:
            return pos
        return (int(round(pos[0] * s)), int(round(pos[1] * s)))

    def mouse_pos(self) -> Tuple[int, int]:
        return self.to_render_pos(pygame.mouse.get_pos())

    def make_scroll_view_scaled(self, area_rect: pygame.Rect, content_h: int, offset_px: int) -> ScrollView:
        return make_scroll_view(
            area_rect=area_rect,
            content_h=content_h,
            offset_px=offset_px,
            scrollbar_w=self.px(SCROLLBAR_W),
            scrollbar_gap=self.px(SCROLLBAR_GAP),
            min_thumb_h=self.px(SCROLLBAR_MIN_THUMB_H),
        )

    def render_size_from_window(self, window_size: Tuple[int, int]) -> Tuple[int, int]:
        s = self.cfg.supersampling
        if s <= 1.0:
            return window_size
        return (
            max(1, int(round(window_size[0] * s))),
            max(1, int(round(window_size[1] * s))),
        )

    def recreate_render_target(self) -> None:
        win_w, win_h = self.display_surface.get_size()

        if self.cfg.supersampling <= 1.0:
            self.screen = self.display_surface
            self.present_surface = None
        else:
            render_w, render_h = self.render_size_from_window((win_w, win_h))
            self.screen = pygame.Surface((render_w, render_h), pygame.SRCALPHA).convert_alpha()
            self.present_surface = pygame.Surface((win_w, win_h)).convert()

    def present(self) -> None:
        if self.screen is self.display_surface:
            pygame.display.flip()
            return

        pygame.transform.smoothscale(self.screen, self.present_surface.get_size(), self.present_surface)
        self.display_surface.blit(self.present_surface, (0, 0))
        pygame.display.flip()

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            self.windowed_size = self.display_surface.get_size()
            desktop_size = pygame.display.get_desktop_sizes()[0]
            self.display_surface = pygame.display.set_mode(
                desktop_size,
                pygame.FULLSCREEN | pygame.DOUBLEBUF,
            )
        else:
            self.display_surface = pygame.display.set_mode(
                self.windowed_size,
                pygame.RESIZABLE | pygame.DOUBLEBUF,
            )

        self.recreate_render_target()
        self.relayout_ui()

        self.draw_loading_screen(
            "Генерация текстур...",
            "Подождите, подготавливаем текстуры для нового режима отображения."
        )
        pygame.event.pump()

        self.preload_visible_textures()
        
    # --------------------------------------------------------
    # Базовые layout helper'ы
    # --------------------------------------------------------
    def compute_layout(self) -> Dict[str, pygame.Rect]:
        w, h = self.screen.get_size()
        margin = max(self.px(10), min(w, h) // 90)
        gap = margin

        toolbar_h = max(self.px(56), self.px(62 * UI_SCALE))
        toolbar_rect = pygame.Rect(margin, margin, w - 2 * margin, toolbar_h)

        map_y = toolbar_rect.bottom + gap
        maps_h = int((h - map_y - margin - gap) * 0.54)
        maps_h = max(self.px(260), maps_h)
        panel_w = (w - 2 * margin - 2 * gap) // 3

        current_rect = pygame.Rect(margin, map_y, panel_w, maps_h)
        candidate_rect = pygame.Rect(current_rect.right + gap, map_y, panel_w, maps_h)
        best_rect = pygame.Rect(candidate_rect.right + gap, map_y, panel_w, maps_h)

        bottom_y = current_rect.bottom + gap
        bottom_h = h - bottom_y - margin

        info_w = int(w * 0.1512 * 0.80)
        log_w = int(w * 0.451 + w * 0.1512 * 0.20)
        routes_w = w - 2 * margin - 2 * gap - info_w - log_w

        min_routes_w = self.px(250)
        if routes_w < min_routes_w:
            deficit = min_routes_w - routes_w
            routes_w = min_routes_w
            log_w = max(self.px(280), log_w - deficit)

        info_rect = pygame.Rect(margin, bottom_y, info_w, bottom_h)
        log_rect = pygame.Rect(info_rect.right + gap, bottom_y, log_w, bottom_h)
        routes_rect = pygame.Rect(log_rect.right + gap, bottom_y, routes_w, bottom_h)

        return {
            "toolbar": toolbar_rect,
            "current": current_rect,
            "candidate": candidate_rect,
            "best": best_rect,
            "info": info_rect,
            "log": log_rect,
            "routes": routes_rect,
        }

    def map_inner_rect(self, rect: pygame.Rect) -> pygame.Rect:
        header_h = max(self.px(56 * UI_SCALE), rect.h // 10)
        pad_x = self.px(18)
        pad_bottom = self.px(18)
        return pygame.Rect(
            rect.x + pad_x,
            rect.y + header_h,
            rect.w - 2 * pad_x,
            rect.h - header_h - pad_bottom,
        )

    def map_node_radius(self, map_rect: pygame.Rect) -> int:
        base = max(self.px(5), min(self.px(10), map_rect.w // 65))
        return base

    def draw_panel_virtual_inner(self, rect: pygame.Rect) -> pygame.Rect:
        return pygame.Rect(
            rect.x + self.px(18),
            rect.y + self.px(50 * UI_SCALE),
            rect.w - self.px(36),
            rect.h - self.px(64 * UI_SCALE),
        )

    def relayout_ui(self) -> None:
        rects = self.compute_layout()
        self.buttons, self.speed_block = self.build_toolbar_controls(rects["toolbar"])

    def get_textured_surface(self, kind: str, size: Tuple[int, int]) -> pygame.Surface:
        w, h = max(1, int(size[0])), max(1, int(size[1]))
        key = (kind, w, h, self.texture_seed)

        if key in self.texture_cache:
            surf = self.texture_cache.pop(key)
            self.texture_cache[key] = surf  # поднять в конец как недавно использованный
            return surf

        if kind == "parchment":
            surf = self._build_parchment_surface((w, h))
        elif kind == "map":
            surf = self._build_map_surface((w, h))
        else:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill(PANEL)

        surf = surf.convert_alpha()
        self.texture_cache[key] = surf
        self.prune_texture_cache()
        return surf

    def _build_parchment_surface(self, size: Tuple[int, int]) -> pygame.Surface:
        w, h = size
        seed = (w * 73856093) ^ (h * 19349663) ^ (self.texture_seed * 83492791) ^ 0x51A7D3
        rs = np.random.RandomState(seed & 0xFFFFFFFF)
        rng = random.Random(seed)

        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)

        def value_noise_2d(width: int, height: int, cell: int) -> np.ndarray:
            gx = max(4, int(math.ceil(width / cell)) + 3)
            gy = max(4, int(math.ceil(height / cell)) + 3)

            grid = rs.rand(gy, gx).astype(np.float32)

            xs = np.arange(width, dtype=np.float32) / float(cell)
            ys = np.arange(height, dtype=np.float32) / float(cell)

            xi = np.floor(xs).astype(np.int32)
            yi = np.floor(ys).astype(np.int32)

            xf = smoothstep(xs - xi)
            yf = smoothstep(ys - yi)

            x0 = xi[None, :]
            x1 = (xi + 1)[None, :]
            y0 = yi[:, None]
            y1 = (yi + 1)[:, None]

            g00 = grid[y0, x0]
            g10 = grid[y0, x1]
            g01 = grid[y1, x0]
            g11 = grid[y1, x1]

            nx0 = g00 * (1.0 - xf[None, :]) + g10 * xf[None, :]
            nx1 = g01 * (1.0 - xf[None, :]) + g11 * xf[None, :]

            return nx0 * (1.0 - yf[:, None]) + nx1 * yf[:, None]

        def norm01(a: np.ndarray) -> np.ndarray:
            return (a - a.min()) / max(1e-9, float(a.max() - a.min()))

        def pulse(frac: np.ndarray, center: float, width: float) -> np.ndarray:
            return np.exp(-((frac - center) ** 2) / (2.0 * width * width))

        def blit_rotated_ellipse(dst: pygame.Surface, color_rgba, center_xy, rx: int, ry: int, angle_deg: float, width: int = 0):
            pad = self.px(8) + width
            tmp = pygame.Surface((2 * rx + 2 * pad, 2 * ry + 2 * pad), pygame.SRCALPHA)
            rect = pygame.Rect(pad, pad, 2 * rx, 2 * ry)
            pygame.draw.ellipse(tmp, color_rgba, rect, width=width)
            rot = pygame.transform.rotozoom(tmp, angle_deg, 1.0)
            dst.blit(rot, rot.get_rect(center=center_xy))

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        n_big = norm01(value_noise_2d(w, h, max(self.px(90), min(w, h) // 3)))
        n_mid = norm01(value_noise_2d(w, h, max(self.px(28), min(w, h) // 8)))
        n_small = norm01(value_noise_2d(w, h, max(self.px(11), min(w, h) // 20)))
        n_micro = norm01(value_noise_2d(w, h, max(self.px(5), min(w, h) // 36)))

        warp1 = norm01(value_noise_2d(w, h, max(self.px(42), min(w, h) // 6)))
        warp2 = norm01(value_noise_2d(w, h, max(self.px(18), min(w, h) // 12)))

        base_noise = 0.56 * n_big + 0.28 * n_mid + 0.16 * n_small
        base_noise = norm01(base_noise)
        pores = np.clip((n_micro - 0.60) * 3.2, 0.0, 1.0) ** 1.55

        ring_period = float(max(self.px(16), h // 16))
        wood_coord = (
            yy
            + self.px0(22.0) * (warp1 - 0.5)
            + self.px0(8.0) * (warp2 - 0.5)
            + 0.035 * xx
        ) / ring_period

        knot_core = np.zeros((h, w), dtype=np.float32)
        knot_rim = np.zeros((h, w), dtype=np.float32)
        knot_bands = np.zeros((h, w), dtype=np.float32)
        knot_shadow = np.zeros((h, w), dtype=np.float32)
        knot_specs = []

        if min(w, h) < self.px(160):
            knot_count = rng.choices([0, 1, 2], weights=[0.20, 0.55, 0.25])[0]
        else:
            knot_count = rng.choices([0, 1, 2, 3], weights=[0.08, 0.34, 0.36, 0.22])[0]

        for _ in range(knot_count):
            cx = rng.randint(w // 7, 6 * w // 7)
            cy = rng.randint(h // 7, 6 * h // 7)

            rx = rng.randint(max(self.px(16), min(w, h) // 28), max(self.px(34), min(w, h) // 12))
            ry = rng.randint(max(self.px(10), min(w, h) // 36), max(self.px(22), min(w, h) // 18))

            ang = rng.uniform(-0.65, 0.65)
            ca = math.cos(ang)
            sa = math.sin(ang)

            dx = xx - cx
            dy = yy - cy

            xr = (ca * dx + sa * dy) / float(rx)
            yr = (-sa * dx + ca * dy) / float(ry)

            r = np.sqrt(xr * xr + yr * yr)
            theta = np.arctan2(yr, xr + 1e-9)

            local = np.exp(-(r ** 2) / (2.0 * 1.15 ** 2))
            wood_coord += local * (1.55 * r + 0.18 * np.sin(3.0 * theta) + 0.10 * np.cos(5.0 * theta))

            knot_coord = 3.8 * r + 0.22 * np.sin(4.0 * theta)
            fk = knot_coord - np.floor(knot_coord)

            kb = (
                1.00 * pulse(fk, 0.18, 0.040) +
                0.70 * pulse(fk, 0.53, 0.060) +
                0.45 * pulse(fk, 0.82, 0.050)
            )
            kb *= local

            core = np.exp(-(r ** 2) / (2.0 * 0.26 ** 2))
            rim = np.exp(-((r - 0.58) ** 2) / (2.0 * 0.08 ** 2))
            shadow = np.exp(-((r - 0.92) ** 2) / (2.0 * 0.13 ** 2))

            knot_core = np.maximum(knot_core, core)
            knot_rim = np.maximum(knot_rim, rim)
            knot_shadow = np.maximum(knot_shadow, shadow * local)
            knot_bands = np.maximum(knot_bands, kb)

            knot_specs.append((cx, cy, rx, ry, math.degrees(ang)))

        frac_main = wood_coord - np.floor(wood_coord)

        dark_lines = (
            1.00 * pulse(frac_main, 0.16, 0.028) +
            0.72 * pulse(frac_main, 0.58, 0.050)
        )
        dark_lines = np.clip(dark_lines, 0.0, 1.0)

        wide_light = 0.5 + 0.5 * np.cos(2.0 * math.pi * (wood_coord - 0.10))
        wide_light = np.clip(wide_light, 0.0, 1.0) ** 1.25

        fine_mod = 0.5 + 0.5 * np.cos(4.0 * math.pi * (wood_coord + 0.07))
        fine_mod = np.clip(fine_mod, 0.0, 1.0) ** 2.1

        cx = 0.5 * w
        cy = 0.5 * h
        center = 1.0 - (((xx - cx) / (1.08 * w)) ** 2 + ((yy - cy) / (1.02 * h)) ** 2)
        center = np.clip(center, 0.0, 1.0)

        noise_small = rs.normal(0.0, 1.0, size=(h, w)).astype(np.float32)

        tone = (
            0.44
            + 0.10 * (base_noise - 0.5)
            + 0.050 * wide_light
            + 0.018 * fine_mod
            - 0.160 * dark_lines
            - 0.105 * knot_bands
            - 0.085 * knot_core
            + 0.030 * knot_rim
            - 0.028 * knot_shadow
            - 0.018 * pores
            + 0.018 * (n_micro - 0.5)
            + 0.020 * center
            + 0.008 * noise_small
        )
        tone = np.clip(tone, 0.0, 1.0)

        dark = np.array(PARCHMENT_DARK, dtype=np.float32)
        light = np.array(PARCHMENT_LIGHT, dtype=np.float32)
        mid = np.array(PARCHMENT_MID, dtype=np.float32)
        base = np.array(PARCHMENT_BASE, dtype=np.float32)

        color = dark[None, None, :] * (1.0 - tone[..., None]) + light[None, None, :] * tone[..., None]
        color = color * 0.80 + mid[None, None, :] * 0.16 + base[None, None, :] * 0.04

        color += wide_light[..., None] * np.array([5, 4, 3], dtype=np.float32)[None, None, :]
        color += fine_mod[..., None] * np.array([2, 2, 1], dtype=np.float32)[None, None, :]

        color -= dark_lines[..., None] * np.array([22, 17, 13], dtype=np.float32)[None, None, :]
        color -= knot_bands[..., None] * np.array([18, 14, 10], dtype=np.float32)[None, None, :]
        color -= knot_core[..., None] * np.array([14, 10, 8], dtype=np.float32)[None, None, :]
        color += knot_rim[..., None] * np.array([5, 4, 3], dtype=np.float32)[None, None, :]
        color -= knot_shadow[..., None] * np.array([8, 6, 5], dtype=np.float32)[None, None, :]
        color -= pores[..., None] * np.array([6, 4, 3], dtype=np.float32)[None, None, :]

        grain_rgb = rs.normal(0.0, 1.2, size=(h, w, 1)).astype(np.float32)
        color = np.clip(color + grain_rgb, 0, 255).astype(np.uint8)

        surf = pygame.surfarray.make_surface(np.transpose(color, (1, 0, 2))).convert_alpha()
        surf = self.soft_blur_surface(surf, factor=2, detail_alpha=108)

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)

        for (cxk, cyk, rxk, ryk, ang_deg) in knot_specs:
            blit_rotated_ellipse(
                overlay,
                (PARCHMENT_DARK[0], PARCHMENT_DARK[1], PARCHMENT_DARK[2], 28),
                (cxk, cyk),
                int(rxk * 1.45),
                int(ryk * 1.20),
                ang_deg,
                width=self.px(2),
            )
            blit_rotated_ellipse(
                overlay,
                (PARCHMENT_LIGHT[0], PARCHMENT_LIGHT[1], PARCHMENT_LIGHT[2], 18),
                (cxk, cyk),
                int(rxk * 1.10),
                int(ryk * 0.92),
                ang_deg,
                width=self.px(1),
            )
            blit_rotated_ellipse(
                overlay,
                (PARCHMENT_DARK[0], PARCHMENT_DARK[1], PARCHMENT_DARK[2], 20),
                (cxk, cyk),
                int(rxk * 0.55),
                int(ryk * 0.48),
                ang_deg,
                width=0,
            )

        scratch_count = max(4, (w * h) // 100000)
        for _ in range(scratch_count):
            start_x = rng.randint(-self.px(20), w // 4)
            start_y = rng.randint(0, h - 1)

            pts = []
            step = max(self.px(16), w // 16)
            cur_y = start_y

            for x in range(start_x, w + step + self.px(20), step):
                cur_y = int(clamp(cur_y + rng.randint(-self.px(2), self.px(2)), 0, h - 1))
                pts.append((x, cur_y))

            alpha_hi = rng.randint(8, 12)
            alpha_lo = max(4, alpha_hi - 4)

            pygame.draw.lines(
                overlay,
                (PARCHMENT_LIGHT[0], PARCHMENT_LIGHT[1], PARCHMENT_LIGHT[2], alpha_hi),
                False,
                pts,
                self.px(1),
            )

            pts_shadow = [(x, min(h - 1, y + self.px(1))) for (x, y) in pts]
            pygame.draw.lines(
                overlay,
                (PARCHMENT_DARK[0], PARCHMENT_DARK[1], PARCHMENT_DARK[2], alpha_lo),
                False,
                pts_shadow,
                self.px(1),
            )

        speck_count = max(140, (w * h) // 2500)
        for _ in range(speck_count):
            x = rng.randrange(w)
            y = rng.randrange(h)
            if rng.random() < 0.52:
                col = PARCHMENT_LIGHT
                alpha = rng.randint(3, 7)
            else:
                col = PARCHMENT_DARK
                alpha = rng.randint(4, 8)
            pygame.draw.circle(
                overlay,
                (col[0], col[1], col[2], alpha),
                (x, y),
                self.px(1),
            )

        line_count = max(10, h // 12)
        for _ in range(line_count):
            y0 = rng.randint(-self.px(8), h + self.px(8))
            pts = []
            step = max(self.px(22), w // 11)
            cur_y = y0
            for x in range(-self.px(10), w + step + self.px(10), step):
                cur_y = int(clamp(cur_y + rng.randint(-self.px(2), self.px(2)), 0, h - 1))
                pts.append((x, cur_y))

            col = PARCHMENT_FIBER if rng.random() < 0.62 else PARCHMENT_DARK
            alpha = rng.randint(5, 9)
            pygame.draw.lines(
                overlay,
                (col[0], col[1], col[2], alpha),
                False,
                pts,
                self.px(1),
            )

        for i in range(self.px(12)):
            pygame.draw.rect(
                overlay,
                (PARCHMENT_STAIN[0], PARCHMENT_STAIN[1], PARCHMENT_STAIN[2], 6 + i * 2),
                pygame.Rect(i, i, max(1, w - 2 * i), max(1, h - 2 * i)),
                width=self.px(1),
                border_radius=self.px(10),
            )

        surf.blit(overlay, (0, 0))
        return surf

    def _build_map_surface(self, size: Tuple[int, int]) -> pygame.Surface:
        w, h = size
        seed = (w * 12582917) ^ (h * 4256249) ^ (self.texture_seed * 961748941) ^ 0x55AA33
        rs = np.random.RandomState(seed & 0xFFFFFFFF)
        rng = random.Random(seed)

        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)

        def norm01(a: np.ndarray) -> np.ndarray:
            return (a - a.min()) / max(1e-9, float(a.max() - a.min()))

        def value_noise_2d(width: int, height: int, cell: int) -> np.ndarray:
            gx = max(4, int(math.ceil(width / cell)) + 3)
            gy = max(4, int(math.ceil(height / cell)) + 3)

            grid = rs.rand(gy, gx).astype(np.float32)

            xs = np.arange(width, dtype=np.float32) / float(cell)
            ys = np.arange(height, dtype=np.float32) / float(cell)

            xi = np.floor(xs).astype(np.int32)
            yi = np.floor(ys).astype(np.int32)

            xf = smoothstep(xs - xi)
            yf = smoothstep(ys - yi)

            x0 = xi[None, :]
            x1 = (xi + 1)[None, :]
            y0 = yi[:, None]
            y1 = (yi + 1)[:, None]

            g00 = grid[y0, x0]
            g10 = grid[y0, x1]
            g01 = grid[y1, x0]
            g11 = grid[y1, x1]

            nx0 = g00 * (1.0 - xf[None, :]) + g10 * xf[None, :]
            nx1 = g01 * (1.0 - xf[None, :]) + g11 * xf[None, :]

            return nx0 * (1.0 - yf[:, None]) + nx1 * yf[:, None]

        def soft_ellipse(dst, color_rgb, alpha, center_xy, rx, ry, layers=6):
            pad = self.px(7)
            tmp = pygame.Surface((rx * 2 + 2 * pad, ry * 2 + 2 * pad), pygame.SRCALPHA)
            tw, th = tmp.get_size()
            for i in range(layers):
                t = i / max(1, layers - 1)
                sx = int(t * rx * 0.38)
                sy = int(t * ry * 0.38)
                a = int(alpha * (1.0 - t) * 0.58)
                rr = pygame.Rect(
                    pad + sx,
                    pad + sy,
                    max(2, tw - 2 * pad - 2 * sx),
                    max(2, th - 2 * pad - 2 * sy),
                )
                pygame.draw.ellipse(tmp, (color_rgb[0], color_rgb[1], color_rgb[2], a), rr)
            dst.blit(tmp, (center_xy[0] - tw // 2, center_xy[1] - th // 2))

        base_scale = max(self.px(42), min(w, h) // 5)
        mid_scale = max(self.px(20), min(w, h) // 9)
        fine_scale = max(self.px(10), min(w, h) // 16)
        micro_scale = max(self.px(5), min(w, h) // 30)

        n1 = value_noise_2d(w, h, base_scale)
        n2 = value_noise_2d(w, h, mid_scale)
        n3 = value_noise_2d(w, h, fine_scale)
        n4 = value_noise_2d(w, h, micro_scale)

        terrain = 0.56 * n1 + 0.28 * n2 + 0.11 * n3 + 0.05 * n4
        terrain = norm01(terrain)

        moist1 = value_noise_2d(w, h, max(self.px(34), min(w, h) // 7))
        moist2 = value_noise_2d(w, h, max(self.px(14), min(w, h) // 13))
        moisture = norm01(0.72 * moist1 + 0.28 * moist2)

        grass_dark = np.array(MAP_GRASS_2, dtype=np.float32)
        grass_light = np.array(MAP_GRASS_1, dtype=np.float32)
        dirt = np.array(MAP_DIRT, dtype=np.float32)
        hill = np.array(MAP_HILL, dtype=np.float32)
        parchment_tint = np.array(PARCHMENT_LIGHT, dtype=np.float32)

        wet_mix = (0.25 + 0.55 * moisture)[..., None]
        color = grass_dark[None, None, :] * (1.0 - wet_mix) + grass_light[None, None, :] * wet_mix

        dirt_mix = np.clip((terrain - 0.52) / 0.30, 0.0, 1.0)
        dirt_mix = (0.14 + 0.58 * dirt_mix + 0.10 * (1.0 - moisture))[..., None]
        color = color * (1.0 - dirt_mix * 0.42) + dirt[None, None, :] * (dirt_mix * 0.42)

        hill_mix = np.clip((terrain - 0.74) / 0.18, 0.0, 1.0)[..., None]
        color = color * (1.0 - hill_mix * 0.55) + hill[None, None, :] * (hill_mix * 0.55)

        gy, gx = np.gradient(terrain)
        shade = 1.0 + (-0.85 * gx - 0.55 * gy)
        shade = np.clip(shade, 0.84, 1.14)
        color = np.clip(color * shade[..., None], 0, 255)

        color = color * 0.93 + parchment_tint[None, None, :] * 0.07

        micro = norm01(n4)
        pores_light = np.clip((micro - 0.52) * 2.4, 0.0, 1.0) ** 1.8
        pores_dark = np.clip((0.48 - micro) * 2.2, 0.0, 1.0) ** 1.7

        color += pores_light[..., None] * np.array([5.0, 4.0, 3.0], dtype=np.float32)[None, None, :]
        color -= pores_dark[..., None] * np.array([4.0, 3.0, 2.5], dtype=np.float32)[None, None, :]

        rgb_grain = rs.normal(0.0, 1.2, size=(h, w, 1)).astype(np.float32)
        color = np.clip(color + rgb_grain, 0, 255)

        dust_field = norm01(value_noise_2d(w, h, max(self.px(26), min(w, h) // 10)))
        dust_mask = np.clip((dust_field - 0.60) * 1.8, 0.0, 1.0) ** 1.6
        color = color * (1.0 - dust_mask[..., None] * 0.05) + parchment_tint[None, None, :] * (dust_mask[..., None] * 0.05)

        color = np.clip(color, 0, 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(color, (1, 0, 2))).convert_alpha()

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)

        dust_blotches = max(4, (w * h) // 90000)
        for _ in range(dust_blotches):
            rx = rng.randint(max(self.px(18), min(w, h) // 20), max(self.px(34), min(w, h) // 10))
            ry = rng.randint(max(self.px(12), min(w, h) // 26), max(self.px(26), min(w, h) // 14))
            x = rng.randrange(w)
            y = rng.randrange(h)
            soft_ellipse(overlay, PARCHMENT_LIGHT, rng.randint(8, 14), (x, y), rx, ry, layers=6)

        speck_count = max(80, (w * h) // 2200)
        for _ in range(speck_count):
            x = rng.randrange(w)
            y = rng.randrange(h)
            if rng.random() < 0.56:
                col = PARCHMENT_LIGHT
                alpha = rng.randint(4, 8)
            else:
                col = PARCHMENT_DARK
                alpha = rng.randint(3, 7)
            pygame.draw.circle(overlay, (col[0], col[1], col[2], alpha), (x, y), self.px(1))

        scratch_count = max(2, (w * h) // 140000)
        for _ in range(scratch_count):
            start_x = rng.randint(-self.px(20), w // 4)
            start_y = rng.randint(0, h - 1)

            pts = []
            step = max(self.px(18), w // 14)
            cur_y = start_y

            for x in range(start_x, w + step + self.px(20), step):
                cur_y = int(clamp(cur_y + rng.randint(-self.px(2), self.px(2)), 0, h - 1))
                pts.append((x, cur_y))

            pygame.draw.lines(
                overlay,
                (PARCHMENT_LIGHT[0], PARCHMENT_LIGHT[1], PARCHMENT_LIGHT[2], rng.randint(4, 7)),
                False,
                pts,
                self.px(1),
            )

        surf.blit(overlay, (0, 0))

        vignette = pygame.Surface((w, h), pygame.SRCALPHA)
        for i in range(self.px(16)):
            pygame.draw.rect(
                vignette,
                (20, 16, 12, 4 + i * 2),
                pygame.Rect(i, i, max(1, w - 2 * i), max(1, h - 2 * i)),
                width=self.px(1),
                border_radius=self.px(8),
            )
        surf.blit(vignette, (0, 0))

        return surf

    def blit_textured_rect(self, rect: pygame.Rect, kind: str, radius: int = 8) -> None:
        tex = self.get_textured_surface(kind, (rect.w, rect.h)).copy()

        mask = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 0))
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)

        tex.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(tex, rect.topleft)

    def soft_blur_surface(self, surf: pygame.Surface, factor: int = 2, detail_alpha: int = 96) -> pygame.Surface:
        w, h = surf.get_size()
        if w <= 2 or h <= 2:
            return surf.copy()

        sw = max(1, w // factor)
        sh = max(1, h // factor)

        small = pygame.transform.smoothscale(surf, (sw, sh))
        blurred = pygame.transform.smoothscale(small, (w, h))

        detail = surf.copy()
        detail.set_alpha(detail_alpha)
        blurred.blit(detail, (0, 0))

        return blurred

    def draw_fantasy_panel_frame(
        self,
        rect: pygame.Rect,
        title: str,
        accent_color: Tuple[int, int, int] = BORDER,
    ) -> pygame.Rect:
        draw_alpha_rounded_rect(self.screen, PANEL_SHADOW, rect.move(self.px(4), self.px(4)), radius=self.px(10))

        pygame.draw.rect(self.screen, WOOD_DARK, rect, border_radius=self.px(10))
        pygame.draw.rect(self.screen, WOOD_MID, rect.inflate(-self.px(4), -self.px(4)), border_radius=self.px(9))
        pygame.draw.rect(self.screen, WOOD_LIGHT, rect.inflate(-self.px(8), -self.px(8)), width=self.px(2), border_radius=self.px(8))

        bronze_rect = rect.inflate(-self.px(12), -self.px(12))
        pygame.draw.rect(self.screen, BRONZE_DARK, bronze_rect, border_radius=self.px(8))
        pygame.draw.rect(self.screen, accent_color, bronze_rect, width=self.px(2), border_radius=self.px(8))

        parchment_rect = bronze_rect.inflate(-self.px(4), -self.px(4))
        self.blit_textured_rect(parchment_rect, "parchment", radius=self.px(7))

        for i in range(self.px(4)):
            pygame.draw.rect(
                self.screen,
                (70, 52, 35, 12 + i * 4),
                parchment_rect.inflate(-2 * i, -2 * i),
                width=self.px(1),
                border_radius=self.px(7),
            )

        pygame.draw.rect(
            self.screen,
            mix_color(BRONZE_LIGHT, PARCHMENT_LIGHT, 0.35),
            parchment_rect,
            width=self.px(1),
            border_radius=self.px(7),
        )

        title_font = self.fonts.get(PANEL_TITLE_SIZE, bold=True)
        tab_w = min(rect.w - self.px(44), max(self.px(150), title_font.size(title)[0] + self.px(40)))
        tab_h = title_font.get_linesize() + self.px(12)

        tab = pygame.Rect(rect.x + self.px(18), rect.y + self.px(18), tab_w, tab_h)

        pygame.draw.rect(self.screen, WOOD_DARK, tab, border_radius=self.px(7))
        pygame.draw.rect(self.screen, BRONZE_DARK, tab.inflate(-self.px(2), -self.px(2)), border_radius=self.px(6))
        inner_tab = tab.inflate(-self.px(4), -self.px(4))
        pygame.draw.rect(self.screen, BRONZE, inner_tab, border_radius=self.px(5))

        shine_rect = pygame.Rect(
            inner_tab.x + self.px(4),
            inner_tab.y + self.px(3),
            inner_tab.w - self.px(8),
            max(self.px(4), inner_tab.h // 3),
        )
        draw_alpha_rounded_rect(self.screen, rgba(BRONZE_LIGHT, 56), shine_rect, radius=self.px(4))

        draw_centered_text(
            self.screen,
            title,
            inner_tab,
            title_font,
            TEXT,
            max_width=inner_tab.w - self.px(16),
        )

        return self.draw_panel_virtual_inner(rect)

    def draw_adventure_map_background(self, rect: pygame.Rect) -> None:
        self.blit_textured_rect(rect, "map", radius=self.px(8))

        pygame.draw.rect(
            self.screen,
            mix_color(BRONZE, MAP_HILL, 0.25),
            rect,
            width=self.px(2),
            border_radius=self.px(8),
        )

        c = mix_color(BRONZE_LIGHT, PARCHMENT_LIGHT, 0.34)
        pad = self.px(10)
        arm = self.px(12)

        pygame.draw.line(self.screen, c, (rect.x + pad, rect.y + pad), (rect.x + pad + arm, rect.y + pad), self.px(1))
        pygame.draw.line(self.screen, c, (rect.x + pad, rect.y + pad), (rect.x + pad, rect.y + pad + arm), self.px(1))

        pygame.draw.line(self.screen, c, (rect.right - pad, rect.y + pad), (rect.right - pad - arm, rect.y + pad), self.px(1))
        pygame.draw.line(self.screen, c, (rect.right - pad, rect.y + pad), (rect.right - pad, rect.y + pad + arm), self.px(1))

        pygame.draw.line(self.screen, c, (rect.x + pad, rect.bottom - pad), (rect.x + pad + arm, rect.bottom - pad), self.px(1))
        pygame.draw.line(self.screen, c, (rect.x + pad, rect.bottom - pad), (rect.x + pad, rect.bottom - pad - arm), self.px(1))

        pygame.draw.line(self.screen, c, (rect.right - pad, rect.bottom - pad), (rect.right - pad - arm, rect.bottom - pad), self.px(1))
        pygame.draw.line(self.screen, c, (rect.right - pad, rect.bottom - pad), (rect.right - pad, rect.bottom - pad - arm), self.px(1))

    # --------------------------------------------------------
    # Loading screen
    # --------------------------------------------------------
    def draw_loading_screen(self, line1: str, line2: Optional[str] = None) -> None:
        self.screen.fill(BG)

        w, h = self.screen.get_size()
        panel_w = min(w - self.px(60), int(w * 0.48))
        panel_h = min(h - self.px(60), int(h * 0.26))
        rect = pygame.Rect((w - panel_w) // 2, (h - panel_h) // 2, panel_w, panel_h)

        pygame.draw.rect(self.screen, PANEL, rect, border_radius=self.px(10))
        pygame.draw.rect(self.screen, BORDER, rect, width=self.px(2), border_radius=self.px(10))

        title_font = self.fonts.get(18, bold=True)
        body_font = self.fonts.get(13)

        draw_text(self.screen, line1, (rect.x + self.px(18), rect.y + self.px(18)), title_font, TEXT, max_width=rect.w - self.px(36))

        if line2:
            y = rect.y + self.px(18) + title_font.get_linesize() + self.px(10)
            for s in wrap_text(body_font, line2, rect.w - self.px(36)):
                draw_text(self.screen, s, (rect.x + self.px(18), y), body_font, MUTED, max_width=rect.w - self.px(36))
                y += body_font.get_linesize() + self.px(4)

        self.present()

    def update_window_title(self) -> None:
        title = f"Визуализатор LNS | день {self.cfg.day} | героев {self.cfg.heroes}"
        pygame.display.set_caption(title)

    def restart(self, new_seed: Optional[int] = None) -> None:
        if new_seed is not None:
            self.seed = new_seed

        self.draw_loading_screen("Перестроение состояния...", f"seed={self.seed}")
        pygame.event.pump()

        self.state = build_app_state(self.full, self.cfg, self.seed)
        self.step_accum = 0.0
        self.log_scroll_px = 0
        self.routes_scroll_px = 0
        self.help_scroll_px = 0
        self.scroll_drag = ScrollDragState()
        self.undo_history.clear()
        self.relayout_ui()
        self.update_window_title()

    # --------------------------------------------------------
    # Undo / step helpers
    # --------------------------------------------------------
    def can_step_back(self) -> bool:
        return len(self.undo_history) > 0

    def push_undo_snapshot(self) -> None:
        if self.state is None or self.state.stepper.stage == "done":
            return
        was_empty = len(self.undo_history) == 0
        self.undo_history.append(self.state.stepper.make_snapshot())
        if was_empty:
            self.relayout_ui()

    def step_forward(self) -> None:
        if self.state is None or self.state.stepper.stage == "done":
            return
        self.push_undo_snapshot()
        self.state.stepper.micro_step()

    def step_back(self) -> None:
        if not self.can_step_back():
            return
        self.autoplay = False
        snap = self.undo_history.pop()
        self.state.stepper.restore_snapshot(snap)
        self.relayout_ui()

    # --------------------------------------------------------
    # Универсальные scroll helper'ы
    # --------------------------------------------------------
    def get_scroll_offset(self, target: str) -> int:
        if target == "log":
            return self.log_scroll_px
        if target == "routes":
            return self.routes_scroll_px
        if target == "help":
            return self.help_scroll_px
        return 0

    def set_scroll_offset(self, target: str, value: int) -> None:
        if target == "log":
            self.log_scroll_px = max(0, int(value))
        elif target == "routes":
            self.routes_scroll_px = max(0, int(value))
        elif target == "help":
            self.help_scroll_px = max(0, int(value))

    def begin_scroll_drag_or_page(self, target: str, view: ScrollView, pos: Tuple[int, int]) -> bool:
        if view.track_rect is None:
            return False

        thumb = scrollbar_thumb_rect(view)
        if thumb is None:
            return False

        if thumb.collidepoint(pos):
            self.scroll_drag.target = target
            self.scroll_drag.grab_offset_y = pos[1] - thumb.y
            return True

        if view.track_rect.collidepoint(pos):
            page_pad = self.px(20)
            if pos[1] < thumb.y:
                self.set_scroll_offset(target, self.get_scroll_offset(target) - view.viewport_h + page_pad)
            elif pos[1] > thumb.bottom:
                self.set_scroll_offset(target, self.get_scroll_offset(target) + view.viewport_h - page_pad)
            return True

        return False

    def update_scroll_drag(self, pos: Tuple[int, int]) -> None:
        if self.scroll_drag.target is None:
            return

        target = self.scroll_drag.target
        rects = self.compute_layout()
        view = self.get_scroll_view_by_target(target, rects)
        if view is None or view.track_rect is None or view.max_offset_px <= 0:
            return

        thumb = scrollbar_thumb_rect(view)
        if thumb is None:
            return

        thumb_h = thumb.h
        travel = max(1, view.track_rect.h - thumb_h)
        new_thumb_y = pos[1] - self.scroll_drag.grab_offset_y
        rel = clamp(new_thumb_y - view.track_rect.y, 0, travel)
        new_offset = int(rel * view.max_offset_px / travel)
        self.set_scroll_offset(target, new_offset)

    def get_scroll_view_by_target(self, target: str, rects: Dict[str, pygame.Rect]) -> Optional[ScrollView]:
        if target == "log":
            return self.get_log_scroll_view(rects["log"])
        if target == "routes":
            return self.get_routes_scroll_view(rects["routes"])[0]
        if target == "help":
            return self.get_help_layout().content_view
        return None

    def apply_wheel(self, pos: Tuple[int, int], dy: int, rects: Dict[str, pygame.Rect]) -> bool:
        wheel_step = self.px(MOUSE_WHEEL_STEP)

        if self.show_help:
            help_layout = self.get_help_layout()
            if help_layout.panel_rect.collidepoint(pos):
                self.help_scroll_px = int(clamp(
                    self.help_scroll_px - dy * wheel_step,
                    0,
                    help_layout.content_view.max_offset_px
                ))
            return True

        targets = [
            ("log", rects["log"]),
            ("routes", rects["routes"]),
        ]
        for target, outer_rect in targets:
            if outer_rect.collidepoint(pos):
                view = self.get_scroll_view_by_target(target, rects)
                if view is not None:
                    self.set_scroll_offset(
                        target,
                        clamp(self.get_scroll_offset(target) - dy * wheel_step, 0, view.max_offset_px)
                    )
                return True
        return False

    def draw_scrolled_items(
        self,
        view: ScrollView,
        items: List[Any],
        height_fn: Callable[[Any, int], int],
        draw_item_fn: Callable[[Any, int, pygame.Rect], None],
    ) -> None:
        mouse = self.mouse_pos()
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(view.area_rect)

        y = view.area_rect.y - view.offset_px
        for idx, item in enumerate(items):
            h = int(height_fn(item, idx))
            row_rect = pygame.Rect(view.area_rect.x, y, view.area_rect.w, h)
            if row_rect.bottom >= view.area_rect.y and row_rect.top <= view.area_rect.bottom:
                draw_item_fn(item, idx, row_rect)
            y += h

        self.screen.set_clip(prev_clip)
        draw_scrollbar(self.screen, view, mouse)

    # --------------------------------------------------------
    # Toolbar
    # --------------------------------------------------------
    def adjust_speed(self, delta: int) -> None:
        self.steps_per_sec = int(clamp(self.steps_per_sec + delta, 1, 60))

    def build_toolbar_controls(self, rect: pygame.Rect) -> Tuple[List[UIButton], UISpeedBlock]:
        buttons = []
        font = self.fonts.get(TOOLBAR_BUTTON_SIZE, bold=True)

        labels = [
            ("play", "Пуск [Sp]"),
            ("pause", "Пауза [Sp]"),
            ("back", "Назад [<]"),
            ("step", "Шаг [>]"),
            ("restart", "Сброс [R]"),
            ("newseed", "Seed [N]"),
            ("ids", "ID [I]"),
            ("help", "Справка [H]"),
        ]

        x = rect.x + self.px(10)
        y = rect.y + self.px(8)
        h = rect.h - self.px(16)
        gap = self.px(8)

        for name, label in labels:
            w = font.size(label)[0] + self.px(28)

            toggled = False
            enabled = True
            if name == "play":
                toggled = self.autoplay
            elif name == "pause":
                toggled = not self.autoplay
            elif name == "ids":
                toggled = self.show_ids
            elif name == "help":
                toggled = self.show_help
            elif name == "back":
                enabled = self.can_step_back()

            btn_rect = pygame.Rect(x, y, w, h)
            buttons.append(UIButton(name=name, label=label, rect=btn_rect, toggled=toggled, enabled=enabled))
            x += w + gap

        value_font = self.fonts.get(TOOLBAR_SPEED_SIZE, bold=True)

        minus_w = self.px(40 * UI_SCALE)
        plus_w = self.px(40 * UI_SCALE)
        value_w = max(self.px(170 * UI_SCALE), value_font.size("Скорость: 60")[0] + self.px(26))

        group_w = minus_w + value_w + plus_w + self.px(8)
        group_h = h
        gx = rect.right - group_w - self.px(10)
        gy = y

        speed_block = UISpeedBlock(
            group_rect=pygame.Rect(gx, gy, group_w, group_h),
            minus_rect=pygame.Rect(gx, gy, minus_w, group_h),
            value_rect=pygame.Rect(gx + minus_w + self.px(4), gy, value_w, group_h),
            plus_rect=pygame.Rect(gx + minus_w + self.px(4) + value_w + self.px(4), gy, plus_w, group_h),
        )

        return buttons, speed_block

    def button_at(self, pos: Tuple[int, int]) -> Optional[UIButton]:
        for btn in self.buttons:
            if btn.rect.collidepoint(pos):
                return btn
        return None

    def speed_hit(self, pos: Tuple[int, int]) -> Optional[str]:
        if self.speed_block is None:
            return None
        if self.speed_block.minus_rect.collidepoint(pos):
            return "minus"
        if self.speed_block.plus_rect.collidepoint(pos):
            return "plus"
        return None

    def handle_button(self, btn: UIButton) -> None:
        if not btn.enabled:
            return

        if btn.name == "play":
            self.autoplay = True
        elif btn.name == "pause":
            self.autoplay = False
        elif btn.name == "back":
            self.step_back()
        elif btn.name == "step":
            self.autoplay = False
            self.step_forward()
        elif btn.name == "restart":
            self.restart(self.seed)
        elif btn.name == "newseed":
            self.restart(int(time.time()) % 10_000_000)
        elif btn.name == "ids":
            self.show_ids = not self.show_ids
        elif btn.name == "help":
            self.show_help = not self.show_help
            self.help_scroll_px = 0

        self.relayout_ui()

    def draw_toolbar(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, TOOLBAR, rect, border_radius=self.px(8))
        pygame.draw.rect(self.screen, BORDER, rect, width=self.px(2), border_radius=self.px(8))

        mouse = self.mouse_pos()
        btn_font = self.fonts.get(TOOLBAR_BUTTON_SIZE, bold=True)
        symbol_font = self.fonts.get(TOOLBAR_SYMBOL_SIZE, bold=True)
        speed_font = self.fonts.get(TOOLBAR_SPEED_SIZE, bold=True)

        for btn in self.buttons:
            draw_button(self.screen, btn, btn_font, mouse_pos=mouse, centered=False)

        if self.speed_block is not None:
            pygame.draw.rect(self.screen, PANEL_ALT, self.speed_block.group_rect, border_radius=self.px(7))
            pygame.draw.rect(self.screen, BUTTON_BORDER, self.speed_block.group_rect, width=self.px(1), border_radius=self.px(7))

            btn_minus = UIButton("__minus__", "-", self.speed_block.minus_rect)
            btn_plus = UIButton("__plus__", "+", self.speed_block.plus_rect)
            btn_value = UIButton("__value__", f"Скорость: {self.steps_per_sec}", self.speed_block.value_rect)

            draw_button(self.screen, btn_minus, symbol_font, mouse_pos=mouse, centered=True)
            draw_button(self.screen, btn_plus, symbol_font, mouse_pos=mouse, centered=True)
            draw_button(self.screen, btn_value, speed_font, mouse_pos=(-9999, -9999), centered=True)

    # --------------------------------------------------------
    # Универсальные panel/helper'ы
    # --------------------------------------------------------
    def draw_panel_box(self, rect: pygame.Rect, title: str) -> pygame.Rect:
        return self.draw_fantasy_panel_frame(rect, title, accent_color=BORDER)

    def draw_tavern_icon(self, center: Tuple[int, int], radius: int) -> None:
        x, y = center

        # radius = радиус БЕЛОГО круга
        white_r = int(radius)

        ring_gold_w = max(self.px(1), white_r // 5)
        ring_black_w = max(self.px(1), white_r // 8)

        outer_r = white_r + ring_gold_w + ring_black_w
        gold_r = white_r + ring_gold_w // 2

        # ----------------------------------------------------
        # Круг таверны: чёрный внешний обод + золотой + белый центр
        # ----------------------------------------------------
        pygame.draw.circle(self.screen, BRONZE_DARK, (x, y), outer_r)
        pygame.draw.circle(self.screen, BRONZE, (x, y), gold_r)
        tavern_fill = mix_color(DEPOT_LABEL, BRONZE_LIGHT, 0.35)
        #tavern_fill = mix_color(DEPOT_LABEL, TOOLTIP_BG, 0.45)
        #tavern_fill = mix_color(tavern_fill, BRONZE_LIGHT, 0.18)
        pygame.draw.circle(self.screen, tavern_fill, (x, y), white_r)

        pygame.draw.circle(self.screen, BRONZE_LIGHT, (x, y), gold_r, max(self.px(1), ring_gold_w // 2))
        pygame.draw.circle(self.screen, BRONZE_DARK, (x, y), outer_r, ring_black_w)

        # ----------------------------------------------------
        # Чёрный схематичный дом внутри круга
        # ----------------------------------------------------
        house_col = (96, 84, 72)

        # Дом на 40% больше и чуть выше
        house_scale = 1.40
        house_shift_y = -int(white_r * 0.10)

        body_w = int(white_r * 0.95 * house_scale)
        body_h = int(white_r * 0.58 * house_scale)
        roof_h = int(white_r * 0.52 * house_scale)

        body_left = x - body_w // 2
        body_top = y + int(white_r * 0.08) + house_shift_y
        body_rect = pygame.Rect(body_left, body_top, body_w, body_h)

        roof_pts = [
            (x - int(body_w * 0.62), body_top + self.px(1)),
            (x, body_top - roof_h),
            (x + int(body_w * 0.62), body_top + self.px(1)),
        ]

        # крыша
        pygame.draw.polygon(self.screen, house_col, roof_pts)

        # корпус
        pygame.draw.rect(self.screen, house_col, body_rect, border_radius=self.px(2))

        # труба
        chim_w = max(self.px(3), int(white_r * 0.14 * house_scale))
        chim_h = max(self.px(6), int(white_r * 0.28 * house_scale))
        chim_rect = pygame.Rect(
            x + int(body_w * 0.18),
            body_top - int(roof_h * 0.52),
            chim_w,
            chim_h,
        )
        pygame.draw.rect(self.screen, house_col, chim_rect, border_radius=self.px(1))

        # светлая дверь
        door_w = max(self.px(4), int(body_w * 0.18))
        door_h = max(self.px(7), int(body_h * 0.52))
        door_rect = pygame.Rect(
            x - door_w // 2,
            body_rect.bottom - door_h,
            door_w,
            door_h,
        )
        pygame.draw.rect(self.screen, tavern_fill, door_rect, border_radius=self.px(1))
        
    def draw_route_segment_soft(
        self,
        surface: pygame.Surface,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        edge_color: Tuple[int, int, int],
        width: int = 4,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2

        pygame.draw.line(surface, color, p1, p2, width)

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return

        nx = -dy / length
        ny = dx / length
        off = width / 2.0

        pygame.draw.aaline(
            surface,
            edge_color,
            (x1 + nx * off, y1 + ny * off),
            (x2 + nx * off, y2 + ny * off),
        )
        pygame.draw.aaline(
            surface,
            edge_color,
            (x1 - nx * off, y1 - ny * off),
            (x2 - nx * off, y2 - ny * off),
        )
        pygame.draw.aaline(surface, color, p1, p2)

    def draw_route_polyline_soft(
        self,
        surface: pygame.Surface,
        pts: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        edge_color: Tuple[int, int, int],
        width: int = 4,
        joint_radius: Optional[int] = None,
    ) -> None:
        if len(pts) < 2:
            return

        for i in range(len(pts) - 1):
            self.draw_route_segment_soft(
                surface=surface,
                p1=pts[i],
                p2=pts[i + 1],
                color=color,
                edge_color=edge_color,
                width=width,
            )

        if joint_radius is None:
            joint_radius = max(1, width // 2)

        for x, y in pts:
            pygame.draw.circle(surface, color, (int(x), int(y)), joint_radius)

    # --------------------------------------------------------
    # Scroll view для ЛОГ и МАРШРУТЫ
    # --------------------------------------------------------
    def get_log_scroll_view(self, rect: pygame.Rect) -> ScrollView:
        inner = self.draw_panel_virtual_inner(rect)
        font = self.fonts.get(LOG_FONT_SIZE, bold=True)
        row_h = font.get_linesize() + self.px(8)
        content_h = len(self.state.stepper.log_entries) * row_h
        view = self.make_scroll_view_scaled(inner, content_h, self.log_scroll_px)
        self.log_scroll_px = view.offset_px
        return view

    def get_routes_scroll_view(self, rect: pygame.Rect) -> Tuple[ScrollView, pygame.Rect]:
        inner = self.draw_panel_virtual_inner(rect)
        title_font = self.fonts.get(ROUTES_SMALL_SIZE, bold=True)
        top_h = title_font.get_linesize() + self.px(10)

        rows_area = pygame.Rect(inner.x, inner.y + top_h, inner.w, inner.h - top_h)
        font = self.fonts.get(ROUTES_FONT_SIZE, bold=True)
        row_h = font.get_linesize() + self.px(12)
        content_h = self.state.day_data.num_heroes * row_h

        view = self.make_scroll_view_scaled(rows_area, content_h, self.routes_scroll_px)
        self.routes_scroll_px = view.offset_px

        title_rect = pygame.Rect(inner.x, inner.y, inner.w, top_h)
        return view, title_rect

    def get_routes_title_hit_rect(self, rect: pygame.Rect) -> pygame.Rect:
        _, title_rect = self.get_routes_scroll_view(rect)
        title_font = self.fonts.get(ROUTES_SMALL_SIZE, bold=True)
        _, title_text = self.current_routes_solution()
        w = min(title_rect.w, title_font.size(title_text)[0] + self.px(6))
        return pygame.Rect(title_rect.x, title_rect.y, w, title_font.get_linesize() + self.px(4))

    # --------------------------------------------------------
    # Help layout / scroll
    # --------------------------------------------------------
    def help_content(self) -> List[Tuple[str, str]]:
        return [
            ("section", "Идея"),
            ("text", "На верхних трёх картах показаны ТЕКУЩЕЕ решение, КАНДИДАТ и ЛУЧШЕЕ."),
            ("text", "LNS сначала строит начальное решение, затем на каждой итерации частично разрушает кандидата (destroy), а потом восстанавливает его (repair)."),
            ("text", "Иногда даже ухудшение может быть принято из-за simulated annealing."),
            ("blank", ""),
            ("section", "Что смотреть"),
            ("text", "- ТЕКУЩЕЕ: принятое решение."),
            ("text", "- КАНДИДАТ: решение, которое сейчас меняется."),
            ("text", "- ЛУЧШЕЕ: лучший найденный результат."),
            ("text", "- ЛОГ: последовательность микрошагов алгоритма."),
            ("text", "- МАРШРУТЫ: компактное представление маршрутов героев."),
            ("blank", ""),
            ("section", "Подсветка"),
            ("text", "- Наведение на строку в ЛОГ подсвечивает связанные мельницы на картах и маршруты."),
            ("text", "- Наведение на круг мельницы на карте подсвечивает маршруты и всплывающие подсказки."),
            ("text", "- Наведение на сегмент маршрута показывает tooltip с длиной сегмента."),
            ("text", "- Наведение на строку в МАРШРУТЫ подсвечивает маршрут героя в выбранном окне решения."),
            ("blank", ""),
            ("section", "Управление"),
            ("text", "Пробел — запуск / пауза"),
            ("text", "Left — шаг назад"),
            ("text", "Right — один микрошаг вперёд"),
            ("text", "Up / Down — быстрее / медленнее"),
            ("text", "F — полноэкранный режим"),
            ("text", "R — сброс с тем же seed"),
            ("text", "N — новый seed"),
            ("text", "I — показать / скрыть ID"),
            ("text", "H — показать / скрыть справку"),
            ("text", "ESC — закрыть справку или выйти"),
        ]

    def build_help_items(self, max_width: int) -> List[HelpRenderItem]:
        section_font = self.fonts.get(HELP_SECTION_SIZE, bold=True)
        body_font = self.fonts.get(HELP_BODY_SIZE, bold=True)
        blank_h = max(self.px(10), self.px(10 * UI_SCALE))

        items: List[HelpRenderItem] = []
        for kind, line in self.help_content():
            if kind == "blank":
                items.append(HelpRenderItem(None, None, HELP_TEXT, blank_h))
            elif kind == "section":
                for wrapped in wrap_text(section_font, line, max_width):
                    items.append(HelpRenderItem(wrapped, section_font, LABEL_ACCENT, section_font.get_linesize() + self.px(4)))
            else:
                for wrapped in wrap_text(body_font, line, max_width):
                    items.append(HelpRenderItem(wrapped, body_font, HELP_TEXT, body_font.get_linesize() + self.px(4)))
        return items

    def get_help_layout(self) -> HelpLayout:
        w, h = self.screen.get_size()

        panel_w = min(w - self.px(40), int(w * 0.80))
        panel_h = min(h - self.px(40), int(h * 0.84))
        panel = pygame.Rect((w - panel_w) // 2, (h - panel_h) // 2, panel_w, panel_h)

        title_font = self.fonts.get(HELP_TITLE_SIZE, bold=True)
        hint_font = self.fonts.get(HELP_HINT_SIZE, bold=True)

        title_lines = wrap_text(title_font, "Справка по визуализатору", panel.w - self.px(36))
        title_h = len(title_lines) * (title_font.get_linesize() + self.px(2)) - self.px(2)
        hint_text = "Закрыть: ESC / H / клик мимо окна"

        content_area = pygame.Rect(
            panel.x + self.px(18),
            panel.y + self.px(16) + title_h + self.px(14),
            panel.w - self.px(36),
            panel.h - self.px(16) - title_h - self.px(14) - self.px(18),
        )

        items = self.build_help_items(content_area.w)
        content_h = sum(item.height for item in items)
        view = self.make_scroll_view_scaled(content_area, content_h, self.help_scroll_px)

        if view.track_rect is not None:
            items = self.build_help_items(view.area_rect.w)
            content_h = sum(item.height for item in items)
            view = self.make_scroll_view_scaled(content_area, content_h, self.help_scroll_px)

        self.help_scroll_px = view.offset_px

        return HelpLayout(
            panel_rect=panel,
            title_font=title_font,
            title_lines=title_lines,
            hint_font=hint_font,
            hint_text=hint_text,
            content_view=view,
            items=items,
        )

    # --------------------------------------------------------
    # Hover helper'ы
    # --------------------------------------------------------
    def panel_title(self, panel_key: str) -> str:
        return {"current": "ТЕКУЩЕЕ", "candidate": "КАНДИДАТ", "best": "ЛУЧШЕЕ"}.get(panel_key, panel_key)

    def solution_by_panel_key(self, panel_key: str) -> Solution:
        if panel_key == "current":
            return self.state.stepper.current
        if panel_key == "candidate":
            return self.state.stepper.candidate
        return self.state.stepper.best

    def current_routes_solution(self) -> Tuple[Solution, str]:
        if self.routes_solution_mode == "current":
            return self.state.stepper.current, "Решение: ТЕКУЩЕЕ"
        if self.routes_solution_mode == "best":
            return self.state.stepper.best, "Решение: ЛУЧШЕЕ"
        return self.state.stepper.candidate, "Решение: КАНДИДАТ"

    def cycle_routes_solution(self) -> None:
        idx = self.routes_solution_order.index(self.routes_solution_mode)
        self.routes_solution_mode = self.routes_solution_order[(idx + 1) % len(self.routes_solution_order)]

    def pick_hovered_log_entry(self, rect: pygame.Rect) -> Optional[LogEntry]:
        view = self.get_log_scroll_view(rect)
        font = self.fonts.get(LOG_FONT_SIZE, bold=True)
        row_h = font.get_linesize() + self.px(8)

        mouse = self.mouse_pos()
        if not view.area_rect.collidepoint(mouse):
            return None

        y = view.area_rect.y - view.offset_px
        for entry in self.state.stepper.log_entries:
            row_rect = pygame.Rect(view.area_rect.x, y, view.area_rect.w, row_h)
            if row_rect.collidepoint(mouse):
                return entry
            y += row_h
        return None

    def pick_hovered_route_row(self, rect: pygame.Rect) -> Optional[int]:
        view, _ = self.get_routes_scroll_view(rect)
        font = self.fonts.get(ROUTES_FONT_SIZE, bold=True)
        row_h = font.get_linesize() + self.px(12)

        mouse = self.mouse_pos()
        if not view.area_rect.collidepoint(mouse):
            return None

        y = view.area_rect.y - view.offset_px
        for hero_idx in range(self.state.day_data.num_heroes):
            row_rect = pygame.Rect(view.area_rect.x, y, view.area_rect.w, row_h)
            if row_rect.collidepoint(mouse):
                return hero_idx
            y += row_h
        return None

    def pick_hovered_graph_object(self, rects: Dict[str, pygame.Rect]) -> Optional[HoveredObjectInfo]:
        mouse = self.mouse_pos()
        best = None

        for panel_key in ["current", "candidate", "best"]:
            map_rect = self.map_inner_rect(rects[panel_key])
            node_r = self.map_node_radius(map_rect)
            if not map_rect.collidepoint(mouse):
                continue

            for ext in self.state.day_data.object_ids_ext:
                pos = self.state.layout.point(ext, map_rect)
                dist = math.hypot(mouse[0] - pos[0], mouse[1] - pos[1])
                if dist <= node_r + self.px(6):
                    info = HoveredObjectInfo(panel_key=panel_key, ext=ext, pos=pos, dist_px=dist)
                    if best is None or info.dist_px < best.dist_px:
                        best = info

        return best

    def pick_hovered_segment(self, rects: Dict[str, pygame.Rect]) -> Optional[HoveredSegmentInfo]:
        mouse = self.mouse_pos()
        best = None

        panels = [
            ("current", "ТЕКУЩЕЕ", self.state.stepper.current),
            ("candidate", "КАНДИДАТ", self.state.stepper.candidate),
            ("best", "ЛУЧШЕЕ", self.state.stepper.best),
        ]

        for panel_key, panel_title, sol in panels:
            panel_rect = rects[panel_key]
            map_rect = self.map_inner_rect(panel_rect)
            if not map_rect.collidepoint(mouse):
                continue

            threshold = max(float(self.px(7)), min(float(self.px(12)), map_rect.w / 120.0))

            for h, route in enumerate(sol.routes):
                if not route:
                    continue

                start_ext = self.state.hero_states_before_day[h].anchor_ext
                ext_chain = [start_ext] + [sol.data.object_id(obj) for obj in route]
                pts = [self.state.layout.point(ext, map_rect) for ext in ext_chain]

                for i in range(len(ext_chain) - 1):
                    p1 = pts[i]
                    p2 = pts[i + 1]
                    dpx = point_segment_distance(mouse[0], mouse[1], p1[0], p1[1], p2[0], p2[1])

                    if dpx <= threshold:
                        seg = HoveredSegmentInfo(
                            panel_key=panel_key,
                            panel_title=panel_title,
                            hero_idx=h,
                            from_ext=ext_chain[i],
                            to_ext=ext_chain[i + 1],
                            length=ext_distance_from_full(self.full, ext_chain[i], ext_chain[i + 1]),
                            p1=p1,
                            p2=p2,
                            dist_px=dpx,
                        )
                        if best is None or seg.dist_px < best.dist_px:
                            best = seg

        return best

    def get_object_tooltip_lines(self, panel_key: str, ext: int) -> List[str]:
        sol = self.solution_by_panel_key(panel_key)
        obj_internal = self.state.day_data.ext_to_int[ext]
        hero_idx = sol.obj_route[obj_internal]

        lines = [f"Мельница {ext}"]

        if hero_idx == -1:
            lines.append("Не назначена")
            return lines

        route = sol.routes[hero_idx]
        pos = sol.obj_pos[obj_internal]
        route_len = len(route)

        prev_ext = self.state.hero_states_before_day[hero_idx].anchor_ext if pos == 0 else sol.data.object_id(route[pos - 1])
        next_ext = sol.data.object_id(route[pos + 1]) if pos + 1 < route_len else None

        lines.append(f"Герой: {hero_idx + 1}")
        lines.append(f"Позиция в маршруте: {pos + 1} / {route_len}")
        lines.append(f"Предыдущая: {format_ext_name(prev_ext)} ({steps_text(ext_distance_from_full(self.full, prev_ext, ext))})")

        if next_ext is None:
            lines.append("Следующая: конец маршрута")
        else:
            lines.append(f"Следующая: {format_ext_name(next_ext)} ({steps_text(ext_distance_from_full(self.full, ext, next_ext))})")

        return lines

    # --------------------------------------------------------
    # События
    # --------------------------------------------------------
    def handle_events(self) -> None:
        rects = self.compute_layout()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.VIDEORESIZE:
                if self.is_fullscreen:
                    continue

                self.begin_resize_loading(event.w, event.h)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_help:
                        self.show_help = False
                        self.relayout_ui()
                    else:
                        self.running = False
                    continue
                elif event.key == pygame.K_SPACE:
                    self.autoplay = not self.autoplay
                elif event.key == pygame.K_LEFT:
                    self.step_back()
                elif event.key == pygame.K_RIGHT:
                    self.autoplay = False
                    self.step_forward()
                elif event.key == pygame.K_UP:
                    self.adjust_speed(+1)
                elif event.key == pygame.K_DOWN:
                    self.adjust_speed(-1)
                elif event.key == pygame.K_r:
                    self.restart(self.seed)
                elif event.key == pygame.K_n:
                    self.restart(int(time.time()) % 10_000_000)
                elif event.key == pygame.K_i:
                    self.show_ids = not self.show_ids
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    if self.show_help:
                        self.help_scroll_px = 0
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()

                self.relayout_ui()

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.scroll_drag = ScrollDragState()

            if event.type == pygame.MOUSEMOTION:
                if self.scroll_drag.target is not None:
                    self.update_scroll_drag(self.to_render_pos(event.pos))

            if event.type == pygame.MOUSEWHEEL:
                self.apply_wheel(self.mouse_pos(), event.y, rects)

            if event.type == pygame.MOUSEBUTTONDOWN:
                render_pos = self.to_render_pos(event.pos)

                if event.button == 1:
                    if self.show_help:
                        help_layout = self.get_help_layout()
                        if help_layout.panel_rect.collidepoint(render_pos):
                            if self.begin_scroll_drag_or_page("help", help_layout.content_view, render_pos):
                                continue
                        else:
                            self.show_help = False
                            self.relayout_ui()
                        continue

                    speed_part = self.speed_hit(render_pos)
                    if speed_part == "minus":
                        self.adjust_speed(-1)
                        continue
                    if speed_part == "plus":
                        self.adjust_speed(+1)
                        continue

                    btn = self.button_at(render_pos)
                    if btn is not None:
                        self.handle_button(btn)
                        continue

                    routes_title_hit = self.get_routes_title_hit_rect(rects["routes"])
                    if routes_title_hit.collidepoint(render_pos):
                        self.cycle_routes_solution()
                        continue

                    if self.begin_scroll_drag_or_page("log", self.get_log_scroll_view(rects["log"]), render_pos):
                        continue
                    if self.begin_scroll_drag_or_page("routes", self.get_routes_scroll_view(rects["routes"])[0], render_pos):
                        continue

                if event.button in (4, 5):
                    wheel_dir = -1 if event.button == 5 else 1
                    self.apply_wheel(render_pos, wheel_dir, rects)

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------
    def update(self, dt: float) -> None:
        self.finalize_resize_if_needed()

        if self.resize_loading:
            return

        if not self.autoplay:
            return

        self.step_accum += dt * self.steps_per_sec
        steps = 0
        while self.step_accum >= 1.0 and steps < 12:
            self.step_forward()
            self.step_accum -= 1.0
            steps += 1
            
    # --------------------------------------------------------
    # Рендер карты и общих элементов
    # --------------------------------------------------------
    def draw_object_label(self, map_rect: pygame.Rect, ext: int, center: Tuple[int, int], node_r: int, font) -> None:
        angle_deg = (ext * 137.507764) % 360.0
        angle = math.radians(angle_deg)

        dx = int(math.cos(angle) * (node_r + self.px(15)))
        dy = int(math.sin(angle) * (node_r + self.px(15)))

        text = str(ext)
        text_surf = font.render(text, True, TEXT)
        text_rect = text_surf.get_rect()
        text_rect.x = center[0] + dx
        text_rect.y = center[1] + dy - text_rect.h // 2

        text_rect.x = clamp(text_rect.x, map_rect.x + self.px(2), map_rect.right - text_rect.w - self.px(2))
        text_rect.y = clamp(text_rect.y, map_rect.y + self.px(2), map_rect.bottom - text_rect.h - self.px(2))

        bg_rect = text_rect.inflate(self.px(8), self.px(4))
        draw_alpha_rounded_rect(self.screen, (15, 13, 11, 170), bg_rect, radius=self.px(4))
        pygame.draw.rect(self.screen, BUTTON_BORDER, bg_rect, width=self.px(1), border_radius=self.px(4))
        self.screen.blit(text_surf, text_rect)

    def draw_candidate_legend(self, map_rect: pygame.Rect, font) -> None:
        line_h = font.get_linesize()
        x = map_rect.x + self.px(30)
        y = max(map_rect.y + self.px(8), map_rect.bottom - (2 * line_h + self.px(18)))

        dx = self.px(6)
        r_small = self.px(5)
        r_removed = self.px(9)
        r_inserted = self.px(10)
        lw = self.px(2)

        pygame.draw.circle(self.screen, TEXT, (x + dx, y + line_h // 2), r_small)
        pygame.draw.circle(self.screen, REMOVED, (x + dx, y + line_h // 2), r_removed, lw)
        draw_outlined_text(
            self.screen,
            "удалён на destroy",
            (x + self.px(20), y - self.px(1)),
            font,
            LEGEND_TEXT,
            outline_color=(18, 14, 11),
        )

        y2 = y + line_h + self.px(5)
        pygame.draw.circle(self.screen, TEXT, (x + dx, y2 + line_h // 2), r_small)
        pygame.draw.circle(self.screen, INSERTED, (x + dx, y2 + line_h // 2), r_inserted, lw)
        draw_outlined_text(
            self.screen,
            "вставлен на repair",
            (x + self.px(20), y2 - self.px(1)),
            font,
            LEGEND_TEXT,
            outline_color=(18, 14, 11),
        )

    def draw_hovered_hero_route(
        self,
        map_rect: pygame.Rect,
        sol: Solution,
        hero_idx: int,
        node_r: int,
    ) -> None:
        if hero_idx < 0 or hero_idx >= sol.data.num_heroes:
            return
        route = sol.routes[hero_idx]
        if not route:
            return

        color = nudge_saturation(HERO_COLORS[hero_idx % len(HERO_COLORS)], 0.22)
        start_ext = self.state.hero_states_before_day[hero_idx].anchor_ext
        ext_chain = [start_ext] + [sol.data.object_id(obj) for obj in route]
        pts = [self.state.layout.point(ext, map_rect) for ext in ext_chain]

        if len(pts) >= 2:
            pygame.draw.lines(self.screen, ROUTE_HOVER, False, pts, self.px(8))
            pygame.draw.lines(self.screen, HOVER_RING, False, pts, self.px(5))
            pygame.draw.lines(self.screen, color, False, pts, self.px(3))

        for ext in ext_chain[1:]:
            pos = self.state.layout.point(ext, map_rect)
            pygame.draw.circle(self.screen, ROUTE_HOVER, pos, node_r + self.px(7), self.px(2))
            pygame.draw.circle(self.screen, HOVER_RING, pos, node_r + self.px(5), self.px(2))

    def draw_map_panel(
        self,
        rect: pygame.Rect,
        panel_key: str,
        title: str,
        sol: Solution,
        border_color: Tuple[int, int, int],
        info_suffix: str = "",
        candidate_removed_ext: Optional[set] = None,
        candidate_inserted_ext: Optional[int] = None,
        hovered_exts: Optional[set] = None,
        hovered_segment_info: Optional[HoveredSegmentInfo] = None,
        hovered_route_hero_idx: Optional[int] = None,
    ) -> None:
        self.draw_fantasy_panel_frame(rect, title, accent_color=border_color)

        title_font = self.fonts.get(PANEL_TITLE_SIZE, bold=True)
        small_font = self.fonts.get(12, bold=True)
        id_font = self.fonts.get(10, bold=True)
        legend_font = self.fonts.get(LEGEND_FONT_SIZE, bold=True)

        q = sol.quality_key()
        stats_text = f"посещено={q[0]} | leftover={q[1]}"
        if info_suffix:
            stats_text += f" | {info_suffix}"

        tab_w = min(rect.w - self.px(44), max(self.px(150), title_font.size(title)[0] + self.px(40)))
        tab_h = title_font.get_linesize() + self.px(12)
        tab_x = rect.x + self.px(18)
        tab_y = rect.y + self.px(22)

        stats_x = tab_x + tab_w + self.px(14)
        stats_y = tab_y + (tab_h - small_font.get_linesize()) // 2 - self.px(1)

        draw_text(
            self.screen,
            stats_text,
            (stats_x, stats_y),
            small_font,
            LABEL_ACCENT,
            max_width=max(self.px(40), rect.right - stats_x - self.px(16)),
        )

        map_rect = self.map_inner_rect(rect)
        self.draw_adventure_map_background(map_rect)

        layout = self.state.layout
        hero_states_before_day = self.state.hero_states_before_day

        assign = {}
        for obj in range(sol.data.object_count):
            r = sol.obj_route[obj]
            if r != -1:
                assign[sol.data.object_id(obj)] = r

        inactive_exts = [ext for ext in sol.data.object_ids_ext if ext not in assign]
        active_exts = [ext for ext in sol.data.object_ids_ext if ext in assign]

        node_r = self.map_node_radius(map_rect)

        # ------------------------------------------------------------------
        # 1. Самый нижний слой: неактивные мельницы
        # ------------------------------------------------------------------
        for ext in inactive_exts:
            pos = layout.point(ext, map_rect)
            pygame.draw.circle(self.screen, UNASSIGNED, pos, node_r)

        # ------------------------------------------------------------------
        # 2. Маршруты
        # ------------------------------------------------------------------
        for h, route in enumerate(sol.routes):
            if not route:
                continue

            base_color = HERO_COLORS[h % len(HERO_COLORS)]
            color = nudge_saturation(base_color, 0.34)
            start_ext = hero_states_before_day[h].anchor_ext
            pts = [layout.point(start_ext, map_rect)] + [layout.point(sol.data.object_id(obj), map_rect) for obj in route]

            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, self.px(3))

        if hovered_route_hero_idx is not None:
            self.draw_hovered_hero_route(map_rect, sol, hovered_route_hero_idx, node_r)

        if hovered_segment_info is not None and hovered_segment_info.panel_key == panel_key:
            pygame.draw.line(self.screen, ROUTE_HOVER, hovered_segment_info.p1, hovered_segment_info.p2, self.px(8))
            pygame.draw.line(self.screen, HOVER_RING, hovered_segment_info.p1, hovered_segment_info.p2, self.px(5))

        # ------------------------------------------------------------------
        # 3. Таверна
        # ------------------------------------------------------------------
        depot_pos = layout.point(0, map_rect)
        tavern_white_r = self.px(13)
        self.draw_tavern_icon(depot_pos, radius=tavern_white_r)

        # ------------------------------------------------------------------
        # 4. Якоря
        # ------------------------------------------------------------------
        for ext, hero_list in layout.anchor_heroes_by_ext.items():
            pos = layout.point(ext, map_rect)
            r = max(self.px(5), min(self.px(9), map_rect.w // 75))
            draw_diamond(self.screen, ANCHOR, pos, r)

            if hovered_exts and ext in hovered_exts:
                draw_diamond(self.screen, HOVER_RING, pos, r + self.px(6), width=self.px(2))

            label = "A" + ",".join(str(x) for x in hero_list[:3])
            if len(hero_list) > 3:
                label += f"+{len(hero_list) - 3}"
            draw_text(self.screen, label, (pos[0] + self.px(6), pos[1] - self.px(14)), small_font, ANCHOR)

        # ------------------------------------------------------------------
        # 5. Активные мельницы поверх неактивных
        # ------------------------------------------------------------------
        for ext in active_exts:
            pos = layout.point(ext, map_rect)
            color = HERO_COLORS[assign[ext] % len(HERO_COLORS)]
            pygame.draw.circle(self.screen, color, pos, node_r)

        # ------------------------------------------------------------------
        # 6. Оверлеи: hover / removed / inserted
        #    Лучше рисовать отдельным проходом, чтобы всегда были сверху
        # ------------------------------------------------------------------
        for ext in sol.data.object_ids_ext:
            pos = layout.point(ext, map_rect)

            if candidate_removed_ext is not None and ext in candidate_removed_ext:
                pygame.draw.circle(self.screen, REMOVED, pos, node_r + self.px(4), self.px(2))

            if candidate_inserted_ext is not None and ext == candidate_inserted_ext:
                pygame.draw.circle(self.screen, INSERTED, pos, node_r + self.px(5), self.px(2))

            if hovered_exts and ext in hovered_exts:
                pygame.draw.circle(self.screen, HOVER_RING, pos, node_r + self.px(8), self.px(3))

        # ------------------------------------------------------------------
        # 7. Подписи id — самый верхний слой узлов
        # ------------------------------------------------------------------
        if self.show_ids:
            for ext in sol.data.object_ids_ext:
                pos = layout.point(ext, map_rect)
                self.draw_object_label(map_rect, ext, pos, node_r, id_font)

        if title == "КАНДИДАТ":
            self.draw_candidate_legend(map_rect, legend_font)

    def build_map_panel_specs(
        self,
        candidate_border: Tuple[int, int, int],
        candidate_suffix: str,
        hovered_exts: set,
        hovered_segment: Optional[HoveredSegmentInfo],
        hovered_route_hero_idx: Optional[int],
        hovered_route_target_panel: Optional[str],
    ) -> List[Dict[str, Any]]:
        stepper = self.state.stepper
        return [
            {
                "key": "current",
                "title": "ТЕКУЩЕЕ",
                "sol": stepper.current,
                "border_color": BORDER,
                "info_suffix": "",
                "candidate_removed_ext": None,
                "candidate_inserted_ext": None,
                "hovered_exts": hovered_exts,
                "hovered_segment_info": hovered_segment,
                "hovered_route_hero_idx": hovered_route_hero_idx if hovered_route_target_panel == "current" else None,
            },
            {
                "key": "candidate",
                "title": "КАНДИДАТ",
                "sol": stepper.candidate,
                "border_color": candidate_border,
                "info_suffix": candidate_suffix,
                "candidate_removed_ext": stepper.highlight_removed_ext,
                "candidate_inserted_ext": stepper.highlight_inserted_ext,
                "hovered_exts": hovered_exts,
                "hovered_segment_info": hovered_segment,
                "hovered_route_hero_idx": hovered_route_hero_idx if hovered_route_target_panel == "candidate" else None,
            },
            {
                "key": "best",
                "title": "ЛУЧШЕЕ",
                "sol": stepper.best,
                "border_color": ACCEPT if stepper.best.visited_count() > 0 else BORDER,
                "info_suffix": "",
                "candidate_removed_ext": None,
                "candidate_inserted_ext": None,
                "hovered_exts": hovered_exts,
                "hovered_segment_info": hovered_segment,
                "hovered_route_hero_idx": hovered_route_hero_idx if hovered_route_target_panel == "best" else None,
            },
        ]

    # --------------------------------------------------------
    # ИНФО
    # --------------------------------------------------------
    def draw_info_kv(self, x: int, y: int, max_w: int, label: str, value: str, label_font, value_font) -> int:
        label_text = f"{label}: "
        label_w = label_font.size(label_text)[0]
        value_w = value_font.size(value)[0]
        line_h = max(label_font.get_linesize(), value_font.get_linesize())

        if label_w + value_w <= max_w:
            draw_text(self.screen, label_text, (x, y), label_font, LABEL_ACCENT, max_width=max_w)
            draw_text(self.screen, value, (x + label_w, y), value_font, VALUE_ACCENT, max_width=max_w - label_w)
            return line_h + self.px(4)

        draw_text(self.screen, label_text, (x, y), label_font, LABEL_ACCENT, max_width=max_w)
        yy = y + line_h + self.px(1)
        for part in wrap_text(value_font, value, max_w - self.px(10)):
            draw_text(self.screen, part, (x + self.px(10), yy), value_font, VALUE_ACCENT, max_width=max_w - self.px(10))
            yy += value_font.get_linesize() + self.px(2)
        return yy - y + self.px(3)

    def draw_info_panel(self, rect: pygame.Rect) -> None:
        inner = self.draw_panel_box(rect, "ИНФО")

        label_font = self.fonts.get(INFO_LABEL_SIZE, bold=True)
        value_font = self.fonts.get(INFO_VALUE_SIZE, bold=True)

        stepper = self.state.stepper
        items = [
            ("Этап", stepper.stage_label()),
            ("Итерация", f"{stepper.iteration} / {self.cfg.iterations if self.cfg.iterations > 0 else 'inf'}"),
            ("Шаг", f"{stepper.substep_in_iter}"),
            ("Destroy", stepper.destroy_op),
            ("Repair", stepper.repair_op),
            ("q", f"{stepper.q_done} / {stepper.q}"),
            ("Температура", f"{stepper.temperature:.3f}"),
            ("Принято", f"{stepper.accepted_moves}"),
            ("Улучшающих", f"{stepper.improving_moves}"),
            ("BEST обновлён", f"{stepper.best_updates}"),
        ]

        x = inner.x
        y = inner.y
        for label, value in items:
            step = self.draw_info_kv(x, y, inner.w, label, value, label_font, value_font)
            y += step
            if y > inner.bottom - self.px(10):
                break

    # --------------------------------------------------------
    # ЛОГ
    # --------------------------------------------------------
    def draw_log_panel(self, rect: pygame.Rect, hovered_exts: set) -> None:
        self.draw_panel_box(rect, "ЛОГ")

        font = self.fonts.get(LOG_FONT_SIZE, bold=True)
        row_h = font.get_linesize() + self.px(8)
        view = self.get_log_scroll_view(rect)
        mouse = self.mouse_pos()
        entries = list(self.state.stepper.log_entries)

        def draw_row(entry: LogEntry, idx: int, row_rect: pygame.Rect) -> None:
            row_hover = row_rect.collidepoint(mouse)

            bg_rect = row_rect.inflate(-self.px(4), -self.px(4))
            bg_rect.y -= self.px(1)

            if row_hover:
                pygame.draw.rect(self.screen, (108, 84, 66), bg_rect, border_radius=self.px(6))
                pygame.draw.rect(self.screen, (176, 142, 102), bg_rect, width=self.px(1), border_radius=self.px(6))

                accent_rect = pygame.Rect(bg_rect.x + self.px(2), bg_rect.y + self.px(3), self.px(4), max(self.px(8), bg_rect.h - self.px(6)))
                pygame.draw.rect(self.screen, LABEL_ACCENT, accent_rect, border_radius=self.px(3))

            prefix = f"[it {entry.iter_no:03d}.{entry.substep:02d}] [{entry.stage}] "
            draw_text(
                self.screen,
                prefix + entry.text,
                (bg_rect.x + self.px(12), bg_rect.y + (bg_rect.h - font.get_linesize()) // 2 - self.px(1)),
                font,
                entry.color,
                max_width=bg_rect.w - self.px(18),
            )

        self.draw_scrolled_items(
            view=view,
            items=entries,
            height_fn=lambda _item, _idx: row_h,
            draw_item_fn=draw_row,
        )

    # --------------------------------------------------------
    # МАРШРУТЫ
    # --------------------------------------------------------
    def draw_routes_panel(self, rect: pygame.Rect, hovered_exts: set, hovered_route_hero_idx: Optional[int]) -> None:
        self.draw_panel_box(rect, "МАРШРУТЫ")
        sol, title = self.current_routes_solution()

        title_font = self.fonts.get(ROUTES_SMALL_SIZE, bold=True)
        font = self.fonts.get(ROUTES_FONT_SIZE, bold=True)

        view, title_rect = self.get_routes_scroll_view(rect)
        mouse = self.mouse_pos()
        title_hit = self.get_routes_title_hit_rect(rect)
        title_color = LABEL_ACCENT if title_hit.collidepoint(mouse) else HELP_TEXT

        draw_text(self.screen, title, (title_rect.x, title_rect.y), title_font, title_color, max_width=title_rect.w)

        row_h = font.get_linesize() + self.px(12)
        hero_items = list(range(self.state.day_data.num_heroes))

        def draw_row(hero_idx: int, idx: int, row_rect: pygame.Rect) -> None:
            color = HERO_COLORS[hero_idx % len(HERO_COLORS)]
            route = sol.routes[hero_idx]
            line = format_route_line(sol, hero_idx, self.state.hero_states_before_day)

            route_exts = {sol.data.object_id(obj) for obj in route}
            start_ext = self.state.hero_states_before_day[hero_idx].anchor_ext

            related = False
            if hovered_exts:
                if route_exts.intersection(hovered_exts) or start_ext in hovered_exts:
                    related = True

            row_hover = row_rect.collidepoint(mouse)
            hero_hover = hovered_route_hero_idx == hero_idx

            bg_rect = row_rect.inflate(-self.px(4), -self.px(4))
            bg_rect.y -= self.px(1)

            if related:
                pygame.draw.rect(self.screen, (86, 62, 50), bg_rect, border_radius=self.px(6))

            if row_hover or hero_hover:
                pygame.draw.rect(self.screen, (110, 82, 64), bg_rect, border_radius=self.px(6))
                accent_rect = pygame.Rect(bg_rect.x + self.px(2), bg_rect.y + self.px(3), self.px(4), max(self.px(8), bg_rect.h - self.px(6)))
                pygame.draw.rect(self.screen, ROUTE_HOVER, accent_rect, border_radius=self.px(3))

            marker_x = view.area_rect.x + self.px(14)
            marker_rect = pygame.Rect(
                marker_x,
                row_rect.y + row_rect.h // 2 - self.px(5),
                self.px(11),
                self.px(11),
            )
            pygame.draw.rect(self.screen, color, marker_rect, border_radius=self.px(2))

            text_x = marker_rect.right + self.px(12)
            text_y = bg_rect.y + (bg_rect.h - font.get_linesize()) // 2 - self.px(1)

            draw_text(
                self.screen,
                line,
                (text_x, text_y),
                font,
                color if route else MUTED,
                max_width=view.area_rect.w - (text_x - view.area_rect.x) - self.px(10),
            )

        self.draw_scrolled_items(
            view=view,
            items=hero_items,
            height_fn=lambda _item, _idx: row_h,
            draw_item_fn=draw_row,
        )

    # --------------------------------------------------------
    # Справка
    # --------------------------------------------------------
    def draw_help_overlay(self) -> None:
        if not self.show_help:
            return

        w, h = self.screen.get_size()
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill(HELP_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        layout = self.get_help_layout()
        panel = layout.panel_rect
        mouse = self.mouse_pos()

        pygame.draw.rect(self.screen, PANEL_ALT, panel, border_radius=self.px(10))
        pygame.draw.rect(self.screen, BORDER, panel, width=self.px(2), border_radius=self.px(10))

        title_x = panel.x + self.px(18)
        title_y = panel.y + self.px(16)

        for line in layout.title_lines:
            draw_text(self.screen, line, (title_x, title_y), layout.title_font, HELP_TEXT, max_width=panel.w - self.px(36))
            title_y += layout.title_font.get_linesize() + self.px(2)

        hint_text_w = layout.hint_font.size(layout.hint_text)[0]
        draw_text(
            self.screen,
            layout.hint_text,
            (panel.right - hint_text_w - self.px(18), panel.y + self.px(18)),
            layout.hint_font,
            HELP_HINT,
        )

        self.draw_scrolled_items(
            view=layout.content_view,
            items=layout.items,
            height_fn=lambda item, _idx: item.height,
            draw_item_fn=lambda item, _idx, row_rect: (
                draw_text(
                    self.screen,
                    item.text,
                    (layout.content_view.area_rect.x, row_rect.y),
                    item.font,
                    item.color,
                    max_width=layout.content_view.area_rect.w,
                ) if item.text is not None and item.font is not None else None
            ),
        )

    # --------------------------------------------------------
    # Tooltip
    # --------------------------------------------------------
    def draw_tooltip(self, lines: List[str], mouse_pos: Tuple[int, int]) -> None:
        if not lines:
            return

        font = self.fonts.get(TOOLTIP_FONT_SIZE, bold=True)
        pad_x = self.px(10)
        pad_y = self.px(8)
        gap_y = self.px(3)

        widths = [font.size(line)[0] for line in lines]
        max_w = max(widths) if widths else self.px(40)
        total_h = len(lines) * font.get_linesize() + (len(lines) - 1) * gap_y

        rect = pygame.Rect(
            mouse_pos[0] + self.px(16),
            mouse_pos[1] + self.px(16),
            max_w + 2 * pad_x,
            total_h + 2 * pad_y,
        )

        sw, sh = self.screen.get_size()
        if rect.right > sw - self.px(8):
            rect.x = mouse_pos[0] - rect.w - self.px(16)
        if rect.bottom > sh - self.px(8):
            rect.y = mouse_pos[1] - rect.h - self.px(16)

        rect.x = clamp(rect.x, self.px(8), sw - rect.w - self.px(8))
        rect.y = clamp(rect.y, self.px(8), sh - rect.h - self.px(8))

        shadow_rect = rect.move(self.px(3), self.px(3))
        pygame.draw.rect(self.screen, (0, 0, 0), shadow_rect, border_radius=self.px(7))
        pygame.draw.rect(self.screen, TOOLTIP_BG, rect, border_radius=self.px(7))
        pygame.draw.rect(self.screen, TOOLTIP_BORDER, rect, width=self.px(1), border_radius=self.px(7))

        y = rect.y + pad_y
        for line in lines:
            draw_text(self.screen, line, (rect.x + pad_x, y), font, TOOLTIP_TEXT, max_width=rect.w - 2 * pad_x)
            y += font.get_linesize() + gap_y

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------
    def render(self) -> None:
        if self.resize_loading:
            self.draw_loading_screen(
                "Генерация текстур...",
                "Отпустите мышь — после завершения изменения размера окна текстуры будут пересобраны."
            )
            return
            
        self.screen.fill(BG)
        rects = self.compute_layout()

        if self.show_help:
            hovered_log_entry = None
            hovered_route_hero_idx = None
            hovered_graph_object = None
            hovered_segment = None
            hovered_exts = set()
            hovered_route_target_panel = None
        else:
            hovered_log_entry = self.pick_hovered_log_entry(rects["log"])
            hovered_route_hero_idx = self.pick_hovered_route_row(rects["routes"])
            hovered_graph_object = self.pick_hovered_graph_object(rects)
            hovered_segment = None if hovered_graph_object is not None else self.pick_hovered_segment(rects)

            hovered_exts = set()
            if hovered_log_entry is not None:
                hovered_exts.update(hovered_log_entry.highlight_exts)
            if hovered_graph_object is not None:
                hovered_exts.add(hovered_graph_object.ext)
            if hovered_segment is not None:
                if hovered_segment.from_ext != 0:
                    hovered_exts.add(hovered_segment.from_ext)
                if hovered_segment.to_ext != 0:
                    hovered_exts.add(hovered_segment.to_ext)

            hovered_route_target_panel = self.routes_solution_mode if hovered_route_hero_idx is not None else None

        self.draw_toolbar(rects["toolbar"])

        stepper = self.state.stepper
        candidate_border = BORDER
        candidate_suffix = stepper.stage

        if stepper.stage in {"destroy", "repair"}:
            candidate_border = IN_PROGRESS
        elif stepper.stage == "post_accept":
            if stepper.last_accept_result is True:
                candidate_border = ACCEPT
                candidate_suffix = "accepted"
            elif stepper.last_accept_result is False:
                candidate_border = REJECT
                candidate_suffix = "rejected"

        panel_specs = self.build_map_panel_specs(
            candidate_border=candidate_border,
            candidate_suffix=candidate_suffix,
            hovered_exts=hovered_exts,
            hovered_segment=hovered_segment,
            hovered_route_hero_idx=hovered_route_hero_idx,
            hovered_route_target_panel=hovered_route_target_panel,
        )
        for spec in panel_specs:
            self.draw_map_panel(
                rect=rects[spec["key"]],
                panel_key=spec["key"],
                title=spec["title"],
                sol=spec["sol"],
                border_color=spec["border_color"],
                info_suffix=spec["info_suffix"],
                candidate_removed_ext=spec["candidate_removed_ext"],
                candidate_inserted_ext=spec["candidate_inserted_ext"],
                hovered_exts=spec["hovered_exts"],
                hovered_segment_info=spec["hovered_segment_info"],
                hovered_route_hero_idx=spec["hovered_route_hero_idx"],
            )

        self.draw_info_panel(rects["info"])
        self.draw_log_panel(rects["log"], hovered_exts=hovered_exts)
        self.draw_routes_panel(rects["routes"], hovered_exts=hovered_exts, hovered_route_hero_idx=hovered_route_hero_idx)
        self.draw_help_overlay()

        if not self.show_help:
            if hovered_graph_object is not None:
                tooltip_lines = self.get_object_tooltip_lines(
                    hovered_graph_object.panel_key,
                    hovered_graph_object.ext,
                )
                self.draw_tooltip(tooltip_lines, self.mouse_pos())

            elif hovered_segment is not None:
                tooltip_lines = [
                    f"Герой {hovered_segment.hero_idx + 1}",
                    f"{format_ext_name(hovered_segment.from_ext)}{ROUTE_ARROW}{format_ext_name(hovered_segment.to_ext)}",
                    f"Длина сегмента: {steps_text(hovered_segment.length)}",
                ]
                self.draw_tooltip(tooltip_lines, self.mouse_pos())

        self.present()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.cfg.fps) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()

        pygame.quit()

# ============================================================
# main
# ============================================================

def main() -> int:
    try:
        cfg = parse_args()
        app = VisualizerApp(cfg)
        app.run()
        return 0
    except Exception as e:
        print(f"{wall_timestamp()} | ERROR | {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
