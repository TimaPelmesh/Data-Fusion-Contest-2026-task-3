#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline на основе LNS-солвера (Data Fusion 2026 — Задача «Герои»).

Запускает LNS с несколькими seeds, берёт лучший результат и сохраняет
submission.csv + summary.txt в папку output/.

Целевой скор: > 295 000.

Использование:
    python pipeline.py                   # стандартный запуск (~15 мин)
    python pipeline.py --fast            # быстрый тест (~5 мин)
    python pipeline.py --heroes 17       # только 17 героев
"""

from __future__ import annotations

import io
import sys

# Принудительно UTF-8 в stdout чтобы не было проблем на Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Константы задачи
# ============================================================
VISIT_COST = 100
HERO_COST = 2500
DAYS = 7

# ============================================================
# Утилиты
# ============================================================

def now_sec() -> float:
    return time.perf_counter()

def wall_ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str) -> None:
    try:
        print(f"{wall_ts()} | {msg}", flush=True)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(f"{wall_ts()} | {safe}", flush=True)

def parse_day_limits(s: str) -> List[float]:
    out = [0.0] * (DAYS + 1)
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) == 1:
        v = float(parts[0])
        for d in range(1, DAYS + 1):
            out[d] = v
        return out
    if len(parts) != DAYS:
        raise ValueError("day-time-limits: нужно 1 или 7 значений")
    for d in range(1, DAYS + 1):
        out[d] = float(parts[d - 1])
    return out


# ============================================================
# Конфигурация
# ============================================================

@dataclass
class Config:
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    heroes: int = 19
    seed: int = 42
    iterations: int = 0          # 0 = до упора по времени
    rcl_size: int = 5
    destroy_frac_min: float = 0.10
    destroy_frac_max: float = 0.35
    temp_start: float = 0.20
    temp_end: float = 0.001
    log_every: int = 200
    day_time_limits: List[float] = field(
        default_factory=lambda: [0.0] * (DAYS + 1)
    )


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

    def dist_by_objid(self, a: int, b: int) -> int:
        return int(self.dist_full[a - 1, b - 1])

    @staticmethod
    def load(data_dir: Path) -> "FullData":
        log(f"Загрузка данных из {data_dir} …")
        heroes_df = pd.read_csv(data_dir / "data_heroes.csv").sort_values("hero_id")
        hero_caps = heroes_df["move_points"].astype(np.int32).tolist()

        objects_df = pd.read_csv(data_dir / "data_objects.csv")
        max_obj = int(objects_df["object_id"].max())
        obj_day = np.zeros(max_obj + 1, dtype=np.int16)
        obj_day[objects_df["object_id"].to_numpy(np.int32)] = \
            objects_df["day_open"].to_numpy(np.int16)

        dist_start_df = pd.read_csv(data_dir / "dist_start.csv")
        dist_start = np.zeros(max_obj + 1, dtype=np.int32)
        dist_start[dist_start_df["object_id"].to_numpy(np.int32)] = \
            dist_start_df["dist_start"].to_numpy(np.int32)

        dist_full = pd.read_csv(data_dir / "dist_objects.csv").to_numpy(
            dtype=np.int32, copy=True
        )
        if dist_full.shape != (max_obj, max_obj):
            raise RuntimeError(f"Матрица расстояний: ожидалось {(max_obj,max_obj)}, получено {dist_full.shape}")

        log(f"Героев: {len(hero_caps)}, мельниц: {max_obj}")
        return FullData(
            hero_caps=hero_caps,
            full_object_count=max_obj,
            object_day_open=obj_day,
            dist_start_by_objid=dist_start,
            dist_full=dist_full,
        )


# ============================================================
# Данные одного дня
# ============================================================

@dataclass
class DayData:
    day: int = 1
    num_heroes: int = 0
    object_count: int = 0
    hero_caps: List[int] = field(default_factory=list)
    object_ids_ext: List[int] = field(default_factory=list)
    start_cost_flat: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int32)
    )
    dist_flat: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.int32)
    )

    def dist(self, a: int, b: int) -> int:
        return int(self.dist_flat[a, b])

    def start_cost(self, h: int, j: int) -> int:
        return int(self.start_cost_flat[h, j])

    def hero_capacity(self, h: int) -> int:
        return int(self.hero_caps[h])

    def object_id(self, j: int) -> int:
        return int(self.object_ids_ext[j])

    def route_cost(self, h: int, route: List[int]) -> int:
        if not route:
            return 0
        total = self.start_cost(h, route[0])
        for i in range(len(route) - 1):
            total += self.dist(route[i], route[i + 1]) + VISIT_COST
        return int(total)

    @staticmethod
    def build_for_day(
        full: FullData,
        day: int,
        heroes_count: int,
        hero_states: List[HeroState],
    ) -> "DayData":
        d = DayData()
        d.day = day
        d.num_heroes = heroes_count
        d.hero_caps = full.hero_caps[:heroes_count]

        ids_arr = (np.where(full.object_day_open[1:] == day)[0] + 1).astype(np.int32)
        d.object_ids_ext = ids_arr.tolist()
        d.object_count = len(d.object_ids_ext)

        if d.object_count > 0:
            idx = ids_arr - 1
            d.dist_flat = full.dist_full[np.ix_(idx, idx)].copy()
        else:
            d.dist_flat = np.zeros((0, 0), dtype=np.int32)

        d.start_cost_flat = np.zeros((d.num_heroes, d.object_count), dtype=np.int32)
        for h in range(d.num_heroes):
            hs = hero_states[h]
            for j in range(d.object_count):
                oe = d.object_ids_ext[j]
                if day == 1:
                    base = int(full.dist_start_by_objid[oe])
                    carry = 0
                else:
                    base = (int(full.dist_start_by_objid[oe])
                            if hs.anchor_ext == 0
                            else full.dist_by_objid(hs.anchor_ext, oe))
                    carry = hs.carry_discount
                d.start_cost_flat[h, j] = (VISIT_COST if carry >= base
                                           else (base - carry) + VISIT_COST)
        return d


# ============================================================
# Решение одного дня
# ============================================================

class Solution:
    def __init__(self, data: DayData):
        self.data = data
        self.routes: List[List[int]] = [[] for _ in range(data.num_heroes)]
        self.route_costs: List[int] = [0] * data.num_heroes
        self.obj_route: List[int] = [-1] * data.object_count
        self.obj_pos: List[int] = [-1] * data.object_count
        self.assigned_count: int = 0

    @staticmethod
    def empty(data: DayData) -> "Solution":
        return Solution(data)

    def clone(self) -> "Solution":
        o = Solution(self.data)
        o.routes = [r.copy() for r in self.routes]
        o.route_costs = self.route_costs.copy()
        o.obj_route = self.obj_route.copy()
        o.obj_pos = self.obj_pos.copy()
        o.assigned_count = self.assigned_count
        return o

    def assigned(self, obj: int) -> bool:
        return self.obj_route[obj] != -1

    def visited_count(self) -> int:
        return self.assigned_count

    def total_leftover(self) -> int:
        return sum(
            max(0, self.data.hero_capacity(r) - self.route_costs[r])
            for r in range(self.data.num_heroes)
        )

    def total_used(self) -> int:
        return sum(
            min(self.data.hero_capacity(r), self.route_costs[r])
            for r in range(self.data.num_heroes)
        )

    def quality_key(self) -> Tuple[int, int]:
        return self.visited_count(), self.total_leftover()

    def update_index_from(self, r: int, start_pos: int) -> None:
        route = self.routes[r]
        for pos in range(max(0, start_pos), len(route)):
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
            return (self.data.start_cost(r, x) - self.data.start_cost(r, b)
                    + self.data.dist(x, b) + VISIT_COST)
        if pos == n - 1:
            a = route[n - 2]
            return self.data.dist(a, x) + VISIT_COST
        a, b = route[pos - 1], route[pos + 1]
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
            return (self.data.start_cost(r, obj) - self.data.start_cost(r, b)
                    + self.data.dist(obj, b) + VISIT_COST)
        if pos == n:
            return self.data.dist(route[n - 1], obj) + VISIT_COST
        a, b = route[pos - 1], route[pos]
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
                if (not found) or delta < best_delta or (delta == best_delta and pos < best_pos):
                    found = True
                    best_delta = delta
                    best_pos = pos
        return (best_delta, best_pos) if found else None

    def insert(self, obj: int, r: int, pos: int, delta: Optional[int] = None) -> None:
        if self.assigned(obj):
            raise RuntimeError("insert: мельница уже назначена")
        if delta is None:
            delta = self.insertion_delta(r, obj, pos)
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

    def validate_basic(self) -> bool:
        seen = [0] * self.data.object_count
        cnt = 0
        for r in range(self.data.num_heroes):
            ac = self.data.route_cost(r, self.routes[r])
            if ac != self.route_costs[r]:
                return False
            if ac > self.data.hero_capacity(r) + VISIT_COST:
                return False
            for pos, obj in enumerate(self.routes[r]):
                if obj < 0 or obj >= self.data.object_count or seen[obj]:
                    return False
                seen[obj] = 1
                if self.obj_route[obj] != r or self.obj_pos[obj] != pos:
                    return False
                cnt += 1
        return cnt == self.assigned_count


# ============================================================
# LNS-операторы
# ============================================================

class DestroyOp(Enum):
    RANDOM = 0
    WORST = 1

class RepairOp(Enum):
    GREEDY = 0
    REGRET2 = 1


# ============================================================
# LNS-солвер одного дня
# ============================================================

class LNSSolver:
    def __init__(self, data: DayData, cfg: Config, seed: int, day_time_limit: float):
        self.data = data
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.day_time_limit = day_time_limit
        self.iters = 0

    def _randint(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def _rand01(self) -> float:
        return self.rng.random()

    def _temperature(self, progress: float) -> float:
        ts, te = self.cfg.temp_start, self.cfg.temp_end
        if ts <= 0 or te <= 0:
            return 0.0
        progress = min(1.0, max(0.0, progress))
        if abs(ts - te) < 1e-15:
            return ts
        return ts * pow(te / ts, progress)

    def _choose_q(self, sol: Solution) -> int:
        n = sol.visited_count()
        if n <= 0:
            return 0
        lo = max(1, int(math.floor(self.cfg.destroy_frac_min * n)))
        hi = min(n, max(lo, int(math.ceil(self.cfg.destroy_frac_max * n))))
        return self._randint(lo, hi)

    def _accept(self, cand: Solution, cur: Solution, temp: float) -> bool:
        ck, uk = cand.quality_key(), cur.quality_key()
        if ck >= uk:
            return True
        delta = (
            (cand.visited_count() - cur.visited_count()) +
            (cand.total_leftover() - cur.total_leftover()) / 1_000_000.0
        )
        return temp > 0 and self._rand01() < math.exp(delta / temp)

    # --- destroy ---
    def _destroy_random(self, sol: Solution, q: int) -> None:
        objs = [o for o in range(self.data.object_count) if sol.assigned(o)]
        self.rng.shuffle(objs)
        for o in objs[:q]:
            sol.remove_object(o)

    def _destroy_worst(self, sol: Solution, q: int) -> None:
        for _ in range(q):
            if sol.visited_count() == 0:
                break
            cands = sorted(
                [(sol.removal_delta(o), o)
                 for o in range(self.data.object_count) if sol.assigned(o)],
                key=lambda x: -x[0],
            )
            limit = min(len(cands), max(1, self.cfg.rcl_size))
            sol.remove_object(cands[self._randint(0, limit - 1)][1])

    def _destroy(self, sol: Solution, op: DestroyOp, q: int) -> None:
        if q <= 0 or sol.visited_count() == 0:
            return
        if op == DestroyOp.RANDOM:
            self._destroy_random(sol, q)
        else:
            self._destroy_worst(sol, q)

    # --- repair ---
    def _greedy_insert_one(self, sol: Solution) -> bool:
        best = (None, -1, -1, -1)  # (delta, r, pos, obj)
        for obj in range(self.data.object_count):
            if sol.assigned(obj):
                continue
            for r in range(self.data.num_heroes):
                ins = sol.best_insertion_in_route(r, obj)
                if ins is None:
                    continue
                delta, pos = ins
                if (best[0] is None
                        or delta < best[0]
                        or (delta == best[0] and r < best[1])
                        or (delta == best[0] and r == best[1] and pos < best[2])
                        or (delta == best[0] and r == best[1] and pos == best[2] and obj < best[3])):
                    best = (delta, r, pos, obj)
        if best[0] is None:
            return False
        sol.insert(best[3], best[1], best[2], best[0])
        return True

    def _repair_greedy(self, sol: Solution) -> None:
        while self._greedy_insert_one(sol):
            pass

    def _regret2_insert_one(self, sol: Solution) -> bool:
        BIG_M = 1_000_000
        chosen = None
        best_regret = -(10 ** 18)
        for obj in range(self.data.object_count):
            if sol.assigned(obj):
                continue
            b1 = b2 = 10 ** 18
            best_r = best_pos = -1
            for r in range(self.data.num_heroes):
                ins = sol.best_insertion_in_route(r, obj)
                if ins is None:
                    continue
                d, p = ins
                if d < b1:
                    b2 = b1
                    b1 = d
                    best_r, best_pos = r, p
                elif d < b2:
                    b2 = d
            if best_r == -1:
                continue
            if b2 >= 10 ** 18:
                b2 = b1 + BIG_M
            regret = b2 - b1
            if (chosen is None
                    or regret > best_regret
                    or (regret == best_regret and b1 < chosen[0])
                    or (regret == best_regret and b1 == chosen[0] and obj < chosen[3])):
                best_regret = regret
                chosen = (b1, best_r, best_pos, obj)
        if chosen is None:
            return False
        sol.insert(chosen[3], chosen[1], chosen[2], chosen[0])
        return True

    def _repair_regret2(self, sol: Solution) -> None:
        while self._regret2_insert_one(sol):
            pass

    def _repair(self, sol: Solution, op: RepairOp) -> None:
        if op == RepairOp.GREEDY:
            self._repair_greedy(sol)
        else:
            self._repair_regret2(sol)

    # --- main loop ---
    def solve(self, deadline: float) -> Solution:
        if self.data.object_count == 0:
            return Solution.empty(self.data)

        cur = Solution.empty(self.data)
        self._repair_greedy(cur)
        best = cur.clone()
        log(
            f"[день {self.data.day}] init  | visited={cur.visited_count():<3} "
            f"| leftover={cur.total_leftover():<6}"
        )

        start = now_sec()
        it = 0
        while now_sec() < deadline and (self.cfg.iterations <= 0 or it < self.cfg.iterations):
            it += 1
            elapsed = now_sec() - start
            progress = min(1.0, elapsed / max(1e-9, self.day_time_limit))
            temp = self._temperature(progress)

            d_op = DestroyOp.RANDOM if self._rand01() < 0.5 else DestroyOp.WORST
            r_op = RepairOp.GREEDY if self._rand01() < 0.5 else RepairOp.REGRET2

            cand = cur.clone()
            q = self._choose_q(cand)
            self._destroy(cand, d_op, q)
            self._repair(cand, r_op)

            if self._accept(cand, cur, temp):
                cur = cand
                if cur.quality_key() > best.quality_key():
                    best = cur.clone()

            if it % self.cfg.log_every == 0:
                log(
                    f"[день {self.data.day}] iter={it:<6} "
                    f"| best=({best.visited_count():>3},{best.total_leftover():>6}) "
                    f"| cur=({cur.visited_count():>3},{cur.total_leftover():>6}) "
                    f"| T={temp:.5f}"
                )

        self.iters = it
        return best


# ============================================================
# Результат и сохранение
# ============================================================

@dataclass
class WeekResult:
    submission_by_hero: List[List[int]] = field(default_factory=list)
    total_visited: int = 0
    total_leftover: int = 0
    total_used: int = 0


def solve_week(full: FullData, cfg: Config) -> WeekResult:
    result = WeekResult(submission_by_hero=[[] for _ in range(cfg.heroes)])
    states: List[HeroState] = [HeroState() for _ in range(cfg.heroes)]

    for day in range(1, DAYS + 1):
        tl = cfg.day_time_limits[day]
        log("-" * 72)
        log(f"DAY {day} | time_limit={int(tl)}s | heroes={cfg.heroes}")

        dd = DayData.build_for_day(full, day, cfg.heroes, states)

        if tl > 0 and dd.object_count > 0:
            run_seed = cfg.seed + day * 10_000_019
            solver = LNSSolver(dd, cfg, run_seed, tl)
            best_day = solver.solve(now_sec() + tl)
        else:
            best_day = Solution.empty(dd)

        if not best_day.validate_basic():
            raise RuntimeError(f"Решение дня {day}: не прошло валидацию")

        for h in range(cfg.heroes):
            for obj in best_day.routes[h]:
                result.submission_by_hero[h].append(dd.object_id(obj))
            if best_day.routes[h]:
                last = best_day.routes[h][-1]
                states[h].anchor_ext = dd.object_id(last)
                states[h].carry_discount = max(0, dd.hero_capacity(h) - best_day.route_costs[h])
            else:
                states[h].carry_discount += dd.hero_capacity(h)

        result.total_visited += best_day.visited_count()
        result.total_leftover += best_day.total_leftover()
        result.total_used += best_day.total_used()

        log(
            f"[день {day}] done | visited={best_day.visited_count():<3} "
            f"| leftover={best_day.total_leftover():<6} | used={best_day.total_used():<6}"
        )

    return result


def save_csv(path: Path, heroes: int, by_hero: List[List[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("hero_id,object_id\n")
        for h in range(heroes):
            for oid in by_hero[h]:
                f.write(f"{h + 1},{oid}\n")


def save_summary(path: Path, r: WeekResult, heroes: int) -> None:
    hcost = heroes * HERO_COST
    reward = r.total_visited * 500
    score = reward - hcost
    lines = [
        f"visited_total={r.total_visited}",
        f"used_moves_total={r.total_used}",
        f"leftover_total={r.total_leftover}",
        f"reward={reward}",
        f"fixed_hero_cost={hcost}",
        f"net_score={score}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return score


# ============================================================
# Multi-seed запуск
# ============================================================

def run_one(full: FullData, cfg: Config) -> Tuple[WeekResult, int]:
    """Одиночный запуск. Возвращает (WeekResult, net_score)."""
    r = solve_week(full, cfg)
    hcost = cfg.heroes * HERO_COST
    return r, r.total_visited * 500 - hcost


def pipeline(
    data_dir: Path,
    output_dir: Path,
    hero_range: Tuple[int, int],
    seeds: List[int],
    day_time_limits: List[float],
) -> None:
    """
    Перебирает hero_count от hero_range[0] до hero_range[1] включительно
    и для каждого числа героев — список seeds.
    Сохраняет лучшее найденное решение.

    Формула скора: score = visited * 500 - heroes * 2500
    Уменьшение числа героев снижает штраф, поэтому стоит проверить 17-19.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full = FullData.load(data_dir)

    h_min, h_max = hero_range
    if h_min < 1 or h_max > full.hero_count() or h_min > h_max:
        raise ValueError(f"Некорректный диапазон героев: {hero_range}")

    best_score: Optional[int] = None
    best_result: Optional[WeekResult] = None
    best_heroes: Optional[int] = None
    best_seed: Optional[int] = None

    for heroes in range(h_min, h_max + 1):
        for seed in seeds:
            log("=" * 72)
            log(f"heroes={heroes} | seed={seed}")
            log("=" * 72)

            cfg = Config(
                data_dir=data_dir,
                output_dir=output_dir,
                heroes=heroes,
                seed=seed,
                day_time_limits=day_time_limits,
            )

            result, score = run_one(full, cfg)
            log(f"-> heroes={heroes} seed={seed}: score={score}  visited={result.total_visited}")

            if best_score is None or score > best_score:
                best_score = score
                best_result = result
                best_heroes = heroes
                best_seed = seed
                log(f"[+] New best score: {score}  (heroes={heroes}, seed={seed})")

                # Сохраняем лучшее решение сразу
                save_csv(output_dir / "submission.csv", heroes, result.submission_by_hero)
                save_summary(output_dir / "summary.txt", result, heroes)

    log("=" * 72)
    log(f"RESULT: best_seed={best_seed}, best_heroes={best_heroes}, score={best_score}")
    log(f"        visited={best_result.total_visited}")
    log(f"        submission -> {output_dir / 'submission.csv'}")
    log("=" * 72)


# ============================================================
# CLI
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(
        description="LNS pipeline для задачи «Герои» (Data Fusion 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Стандартный запуск (~50 мин, воспроизводит скор ~295000+):
  python pipeline.py

  # Быстрый тест (~4 мин, скор ~292000+):
  python pipeline.py --fast

  # Только seed=42, 19 героев, без перебора:
  python pipeline.py --heroes-range 19,19 --seeds 42

  # Полный перебор 17-19 героев x 3 seeds (~2.5 ч):
  python pipeline.py --heroes-range 17,19 --seeds 42,137,999
        """,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Папка с данными (по умолчанию ./data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Папка для результатов (по умолчанию ./output)",
    )
    p.add_argument(
        "--heroes-range",
        type=str,
        default="17,19",
        help=(
            "Диапазон числа героев для перебора: min,max включительно "
            "(по умолчанию 17,19 — перебирает 17, 18, 19)"
        ),
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Список seeds через запятую (по умолчанию 42)",
    )
    p.add_argument(
        "--day-time-limits",
        type=str,
        default="1800,180,180,180,180,180,180",
        help=(
            "Лимиты времени по дням в секундах: одно число (для всех дней) "
            "или 7 чисел через запятую. "
            "По умолчанию: 1800,180,180,180,180,180,180 (~50 мин суммарно)"
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Быстрый тест: 60s день 1, 30s остальные, seed=42, heroes=19 (~4 мин)",
    )
    args = p.parse_args()

    if args.fast:
        seeds = [42]
        day_limits = parse_day_limits("60,30,30,30,30,30,30")
        hero_range = (19, 19)
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        day_limits = parse_day_limits(args.day_time_limits)
        hr_parts = [x.strip() for x in args.heroes_range.split(",")]
        if len(hr_parts) == 1:
            h = int(hr_parts[0])
            hero_range = (h, h)
        elif len(hr_parts) == 2:
            hero_range = (int(hr_parts[0]), int(hr_parts[1]))
        else:
            raise ValueError(f"--heroes-range: ожидается 'min,max', получено '{args.heroes_range}'")

    log("=" * 72)
    log("LNS PIPELINE — Data Fusion 2026 Heroes")
    log(f"heroes-range: {hero_range[0]}..{hero_range[1]}")
    log(f"seeds: {seeds}")
    log(f"day-time-limits: {[int(day_limits[d]) for d in range(1, DAYS+1)]}")
    log("=" * 72)

    try:
        pipeline(
            data_dir=args.data_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            hero_range=hero_range,
            seeds=seeds,
            day_time_limits=day_limits,
        )
        return 0
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
