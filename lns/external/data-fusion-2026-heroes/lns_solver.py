#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    ===============================================================
    Решение задачи маршрутизации героев с временными окнами
    на основе LNS (Large Neighborhood Search)
    ===============================================================

    Идея задачи
    -----------
    У нас есть:
      - таверна, где можно нанимать героев;
      - герои с разным запасом очков хода;
      - мельницы, каждая из которых даёт золото только в свой день недели;
      - матрица расстояний между мельницами и расстояния от таверны.

    Нужно построить маршруты героев так, чтобы за неделю успеть посетить
    как можно больше мельниц в правильные дни.

    В этой версии мы используем упрощённую посуточную модель:
      - неделя разбивается на 7 независимых дневных подзадач;
      - для каждого дня рассматриваются только мельницы, которые открываются
        в этот день;
      - между днями переносится состояние героя:
            * возле какой мельницы он закончил день;
            * сколько "запаса движения" можно учесть на следующий день.

    ---------------------------------------------------------------
    Почему здесь LNS
    ---------------------------------------------------------------
    Задача маршрутизации с временными окнами является NP-трудной,
    поэтому полный перебор невозможен.

    Вместо этого используется эвристика LNS:
      1. строим начальное решение;
      2. многократно:
           - разрушаем часть текущего решения (destroy),
           - восстанавливаем его другим способом (repair),
           - иногда принимаем даже ухудшение решения
             (simulated annealing), чтобы не застревать
             в локальном оптимуме;
      3. запоминаем лучшее найденное решение.

    ---------------------------------------------------------------
    Как устроено это решение
    ---------------------------------------------------------------
    1. Загрузка данных
       Из CSV читаются:
         - герои и их move_points,
         - мельницы и день их открытия,
         - расстояния от таверны,
         - полная матрица расстояний между мельницами.

    2. Посуточная декомпозиция
       Для каждого дня строится DayData:
         - список мельниц этого дня,
         - компактная матрица расстояний между ними,
         - стоимость стартового перехода героя к первой мельнице.

    3. Представление решения
       Solution хранит:
         - маршруты героев,
         - стоимости маршрутов,
         - быстрые индексы, где находится каждая мельница.

    4. Destroy-операторы
         - RANDOM : случайно удаляет часть мельниц;
         - WORST  : удаляет "неудобные" мельницы, которые сильнее всего
                    увеличивают стоимость маршрутов.

    5. Repair-операторы
         - GREEDY  : вставляет мельницу туда, где это дешевле всего;
         - REGRET2 : раньше вставляет "хрупкие" мельницы, которые потом
                     может быть трудно вставить.

    6. Принятие решений
       Если новое решение лучше - принимаем всегда.
       Если хуже - можем принять с некоторой вероятностью.
       Вероятность зависит от температуры simulated annealing,
       которая со временем уменьшается.

    ---------------------------------------------------------------
    Что оптимизируется
    ---------------------------------------------------------------
    Внутри дневного LNS качество решения сравнивается по ключу:
        (число посещённых мельниц, суммарный leftover)

    То есть:
      - сначала хотим посетить больше мельниц;
      - при равенстве предпочитаем решение, где у героев остаётся
        больше очков хода.

    ---------------------------------------------------------------
    Формат выхода
    ---------------------------------------------------------------
    Программа сохраняет:
      - submission.csv  : итоговые маршруты в формате hero_id,object_id
      - summary.txt     : краткую статистику по запуску

    Пример запуска:
      python lns_solver.py \
        --data-dir data \
        --output-dir out \
        --heroes 20 \
        --day-time-limits 60 \
        --seed 42 \
        --iterations 0 \
        --rcl-size 5 \
        --destroy-frac-min 0.10 \
        --destroy-frac-max 0.35 \
        --temp-start 0.20 \
        --temp-end 0.001 \
        --log-every 100
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Глобальные константы задачи
# ============================================================

# Стоимость посещения мельницы в очках хода.
VISIT_COST = 100

# Стоимость найма одного героя.
HERO_COST = 2500

# Количество дней в неделе.
DAYS = 7


# ============================================================
# Вспомогательные функции
# ============================================================

# Текущее монотонное время в секундах.
# Используется только для контроля лимита времени алгоритма.
def now_sec() -> float:
    return time.perf_counter()


# Текущее время в виде строки.
# Нужно только для красивых логов.
def wall_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Простой вывод строки в лог.
def log_msg(msg: str) -> None:
    print(f"{wall_timestamp()} | {msg}", flush=True)


# Парсим лимиты времени по дням.
# Можно дать одно число - тогда оно применяется ко всем дням.
# Или 7 чисел через запятую - отдельно на каждый день.
def parse_day_time_limits(s: str) -> List[float]:
    out = [0.0] * (DAYS + 1)
    parts = [x.strip() for x in s.split(",") if x.strip()]

    if len(parts) == 1:
        x = float(parts[0])
        for d in range(1, DAYS + 1):
            out[d] = x
        return out

    if len(parts) != DAYS:
        raise RuntimeError("--day-time-limits должен содержать либо 1 число, либо 7 чисел")

    for d in range(1, DAYS + 1):
        out[d] = float(parts[d - 1])

    return out


# ============================================================
# Конфигурация запуска
# ============================================================

@dataclass
class Config:
    # Папка, где лежат входные данные.
    data_dir: Path = Path("data")

    # Папка, куда сохраняем результаты работы.
    output_dir: Path = Path("out_lns")

    # Сколько первых героев использовать.
    # В этой версии считаем, что парк героев фиксирован заранее.
    heroes: int = 17

    # Начальное зерно генератора случайных чисел.
    # От него зависит воспроизводимость результата.
    seed: int = 42

    # Максимум итераций LNS на один день.
    # Если 0, работаем до исчерпания времени.
    iterations: int = 0

    # Размер restricted candidate list в destroy_worst.
    # Мы сортируем кандидатов по "выгоде удаления",
    # а потом случайно выбираем одного из лучших rcl_size кандидатов.
    rcl_size: int = 5

    # Минимальная доля мельниц, которую можно удалить на фазе destroy.
    destroy_frac_min: float = 0.10

    # Максимальная доля мельниц, которую можно удалить на фазе destroy.
    destroy_frac_max: float = 0.35

    # Начальная температура simulated annealing.
    # Большая температура => чаще принимаем ухудшения в начале.
    temp_start: float = 0.20

    # Конечная температура simulated annealing.
    # Маленькая температура => в конце поиск становится почти жадным.
    temp_end: float = 0.001

    # Как часто печатать лог по итерациям.
    log_every: int = 100

    # Лимит времени для каждого дня.
    # day_time_limits[1] ... day_time_limits[7]
    day_time_limits: List[float] = field(default_factory=lambda: [0.0] * (DAYS + 1))


# Ручной разбор аргументов запуска.
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="LNS solver for hero routing with time windows (Data Fusion 2026 - Heroes Task)"
    )

    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("out_lns"))
    parser.add_argument("--heroes", type=int, default=17)
    parser.add_argument("--day-time-limits", type=str, default="1800,180,180,180,180,180,180")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=0, help="0 = until time limit")
    parser.add_argument("--rcl-size", type=int, default=5)
    parser.add_argument("--destroy-frac-min", type=float, default=0.10)
    parser.add_argument("--destroy-frac-max", type=float, default=0.35)
    parser.add_argument("--temp-start", type=float, default=0.20)
    parser.add_argument("--temp-end", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        heroes=args.heroes,
        seed=args.seed,
        iterations=args.iterations,
        rcl_size=args.rcl_size,
        destroy_frac_min=args.destroy_frac_min,
        destroy_frac_max=args.destroy_frac_max,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        log_every=args.log_every,
        day_time_limits=parse_day_time_limits(args.day_time_limits),
    )

    if cfg.heroes <= 0:
        raise RuntimeError("--heroes должно быть > 0")
    if cfg.rcl_size <= 0:
        raise RuntimeError("--rcl-size должно быть > 0")
    if cfg.destroy_frac_min <= 0 or cfg.destroy_frac_max <= 0 or cfg.destroy_frac_min > cfg.destroy_frac_max:
        raise RuntimeError("Некорректные destroy-frac параметры")
    if cfg.log_every <= 0:
        raise RuntimeError("--log-every должно быть > 0")

    return cfg


# ============================================================
# Полные данные задачи
# ============================================================

@dataclass
class HeroState:
    # Возле какой мельницы герой закончил предыдущий день.
    # 0 означает, что в нашей модели он стартует из таверны.
    anchor_ext: int = 0

    # "Скидка" на стартовый переход следующего дня.
    # Это упрощённый способ переноса неиспользованного движения между днями
    # в day-by-day модели.
    carry_discount: int = 0


@dataclass
class FullData:
    # Дневной запас очков хода для героев.
    # hero_id отдельно не храним - считаем, что используем героев 1..K.
    hero_caps: List[int]

    # Число мельниц в исходной задаче.
    full_object_count: int

    # object_id -> день открытия объекта.
    object_day_open: np.ndarray

    # Расстояние от таверны до мельницы.
    dist_start_by_objid: np.ndarray

    # Полная матрица расстояний между мельницами.
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

        log_msg("Загрузка данных из: " + str(data_dir))

        # ----------------------------------------------------
        # 1. Герои
        # ----------------------------------------------------
        heroes_df = pd.read_csv(heroes_path)
        heroes_df = heroes_df.sort_values("hero_id")
        hero_caps = heroes_df["move_points"].astype(np.int32).tolist()

        # ----------------------------------------------------
        # 2. Мельницы и дни открытия
        # ----------------------------------------------------
        objects_df = pd.read_csv(objects_path)
        max_obj_id = int(objects_df["object_id"].max())

        object_day_open = np.zeros(max_obj_id + 1, dtype=np.int16)
        obj_ids = objects_df["object_id"].to_numpy(dtype=np.int32)
        day_vals = objects_df["day_open"].to_numpy(dtype=np.int16)
        object_day_open[obj_ids] = day_vals

        # ----------------------------------------------------
        # 3. Расстояния от таверны
        # ----------------------------------------------------
        dist_start_df = pd.read_csv(dist_start_path)
        dist_start_by_objid = np.zeros(max_obj_id + 1, dtype=np.int32)

        ds_ids = dist_start_df["object_id"].to_numpy(dtype=np.int32)
        ds_vals = dist_start_df["dist_start"].to_numpy(dtype=np.int32)
        dist_start_by_objid[ds_ids] = ds_vals

        # ----------------------------------------------------
        # 4. Матрица расстояний между мельницами
        # ----------------------------------------------------
        # В pandas это читается проще, чем построчный разбор.
        # Предполагаем, что файл - обычная числовая таблица 700x700 с header.
        dist_full = pd.read_csv(dist_matrix_path).to_numpy(dtype=np.int32, copy=True)

        if dist_full.shape != (max_obj_id, max_obj_id):
            raise RuntimeError(
                f"Неверный размер матрицы в {dist_matrix_path}: "
                f"{dist_full.shape}, ожидалось {(max_obj_id, max_obj_id)}"
            )

        log_msg(f"Героев: {len(hero_caps)}")
        log_msg(f"Мельниц: {max_obj_id}")

        return FullData(
            hero_caps=hero_caps,
            full_object_count=max_obj_id,
            object_day_open=object_day_open,
            dist_start_by_objid=dist_start_by_objid,
            dist_full=dist_full,
        )


# ============================================================
# Данные одного дня
# ============================================================

# DayData - это посуточный "срез" задачи.
#
# Мы решаем не сразу всю неделю одним огромным маршрутом,
# а по дням: отдельно день 1, отдельно день 2 и т.д.
#
# Для этого на каждый день строим компактную подзадачу:
# - берём только мельницы, которые открываются в этот день;
# - строим расстояния между ними;
# - считаем стоимость старта каждого героя к каждой такой мельнице.
@dataclass
class DayData:
    day: int = 1
    num_heroes: int = 0
    object_count: int = 0

    # Запас хода каждого героя в этот день.
    hero_caps: List[int] = field(default_factory=list)

    # Перевод внутреннего индекса мельницы во внешний object_id.
    object_ids_ext: List[int] = field(default_factory=list)

    # Стоимость "поставить мельницу первой" для героя.
    start_cost_flat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))

    # Матрица расстояний между мельницами этого дня.
    dist_flat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))

    def dist(self, a: int, b: int) -> int:
        return int(self.dist_flat[a, b])

    def start_cost(self, hero_idx: int, obj_idx: int) -> int:
        return int(self.start_cost_flat[hero_idx, obj_idx])

    def hero_capacity(self, h: int) -> int:
        return int(self.hero_caps[h])

    def object_id(self, obj: int) -> int:
        return int(self.object_ids_ext[obj])

    def route_cost(self, hero_idx: int, route: List[int]) -> int:
        if not route:
            return 0

        total = self.start_cost(hero_idx, route[0])
        for i in range(len(route) - 1):
            total += self.dist(route[i], route[i + 1]) + VISIT_COST
        return int(total)

    # --------------------------------------------------------
    # build_for_day
    # --------------------------------------------------------
    #
    # Эта функция строит "локальную" задачу для одного дня.
    #
    # Что именно она делает:
    #
    # 1. Берёт только первых heroes_count героев.
    # 2. Оставляет только те мельницы, которые открываются в день day.
    # 3. Переиндексирует эти мельницы во внутренние индексы 0..object_count-1.
    # 4. Строит маленькую матрицу расстояний между мельницами этого дня.
    # 5. Для каждого героя и каждой мельницы считает стоимость старта:
    #      - если это 1-й день, стартуем из таверны;
    #      - иначе стартуем от последней мельницы (в маршруте) предыдущего дня;
    #      - учитываем carry_discount, то есть накопленный "запас" движения.
    #
    # Зачем всё это нужно:
    #
    # LNS на 700 объектах недели сразу был бы слишком тяжёлым и сложным.
    # Посуточная декомпозиция уменьшает размер задачи:
    # в конкретный день мы смотрим только на мельницы этого дня.
    #
    # Важная идея start_cost:
    #
    # Для первой мельницы маршрута мы не хотим каждый раз отдельно помнить,
    # откуда пришёл герой и сколько движения у него осталось.
    # Поэтому заранее считаем "стоимость поставить мельницу первой" (в маршрут).
    # Это упрощает дальнейшие операции вставки/удаления.
    @staticmethod
    def build_for_day(
        full: FullData,
        day: int,
        heroes_count: int,
        hero_states_before_day: List[HeroState],
    ) -> "DayData":
        if heroes_count > full.hero_count():
            raise RuntimeError("Запрошено героев больше, чем доступно")
        if len(hero_states_before_day) != heroes_count:
            raise RuntimeError("hero_states_before_day.size() != heroes_count")

        data = DayData()
        data.day = day
        data.num_heroes = heroes_count

        # Берём только первых heroes_count героев.
        data.hero_caps = full.hero_caps[:heroes_count]

        # Выбираем только мельницы, которые открываются именно сегодня.
        # Это и есть "дневной набор задач" для LNS.
        object_ids_ext_arr = (np.where(full.object_day_open[1:] == day)[0] + 1).astype(np.int32)
        data.object_ids_ext = object_ids_ext_arr.tolist()
        data.object_count = len(data.object_ids_ext)

        # Строим маленькую матрицу расстояний только между мельницами дня.
        # Это уменьшает объём данных, с которыми работает поиск.
        if data.object_count > 0:
            idx = object_ids_ext_arr - 1
            data.dist_flat = full.dist_full[np.ix_(idx, idx)].copy()
        else:
            idx = np.empty(0, dtype=np.int32)
            data.dist_flat = np.zeros((0, 0), dtype=np.int32)

        # start_cost_flat[h, j]:
        # стоимость поставить мельницу j первой в маршруте героя h.
        data.start_cost_flat = np.zeros((data.num_heroes, data.object_count), dtype=np.int32)

        for h in range(data.num_heroes):
            hs = hero_states_before_day[h]

            for j in range(data.object_count):
                obj_ext = data.object_ids_ext[j]

                base_dist = 0
                carry = 0

                if day == 1:
                    # В первый день герой стартует из таверны.
                    base_dist = int(full.dist_start_by_objid[obj_ext])
                    carry = 0
                else:
                    # В следующие дни стартуем от точки завершения предыдущего дня.
                    if hs.anchor_ext == 0:
                        base_dist = int(full.dist_start_by_objid[obj_ext])
                    else:
                        base_dist = full.dist_by_objid(hs.anchor_ext, obj_ext)
                    carry = hs.carry_discount

                # Если накопленной "скидки" хватает на дорогу,
                # остаётся только VISIT_COST.
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
        # Ссылка на данные конкретного дня.
        # Через неё можно быстро получать расстояния и стартовые стоимости.
        self.data = data

        # routes[r] - маршрут героя r.
        # Внутри маршрута лежат внутренние индексы мельниц дня.
        # Например, если routes[2] = {5, 1, 7}, значит герой 2 посещает сначала
        # мельницу №5 (во внутренней нумерации дня), потом №1, потом №7.
        self.routes: List[List[int]] = [[] for _ in range(data.num_heroes)]

        # route_costs[r] - стоимость маршрута routes[r].
        # Мы кэшируем эти значения, чтобы не пересчитывать стоимость маршрута целиком
        # после каждой маленькой операции.
        self.route_costs: List[int] = [0 for _ in range(data.num_heroes)]

        # obj_route[obj] - номер маршрута, в который назначена мельница.
        # -1 означает, что мельница пока не назначена.
        self.obj_route: List[int] = [-1 for _ in range(data.object_count)]

        # obj_pos[obj] - позиция мельницы внутри маршрута.
        self.obj_pos: List[int] = [-1 for _ in range(data.object_count)]

        # Число мельниц, уже назначенных в решение.
        self.assigned_count: int = 0

    @staticmethod
    def empty(data: DayData) -> "Solution":
        return Solution(data)

    def clone(self) -> "Solution":
        other = Solution(self.data)
        other.routes = [route.copy() for route in self.routes]
        other.route_costs = self.route_costs.copy()
        other.obj_route = self.obj_route.copy()
        other.obj_pos = self.obj_pos.copy()
        other.assigned_count = self.assigned_count
        return other

    def assigned(self, obj: int) -> bool:
        return self.obj_route[obj] != -1

    def visited_count(self) -> int:
        return self.assigned_count

    # Tie-break:
    # если два решения собрали одинаковое число мельниц,
    # мы предпочитаем то, где осталось больше движения.
    def total_leftover(self) -> int:
        total = 0
        for r in range(self.data.num_heroes):
            total += max(0, self.data.hero_capacity(r) - self.route_costs[r])
        return total

    # Сколько движения потрачено по всем героям.
    # Для статистики и анализа.
    def total_used(self) -> int:
        total = 0
        for r in range(self.data.num_heroes):
            total += min(self.data.hero_capacity(r), self.route_costs[r])
        return total

    # --------------------------------------------------------
    # quality_key
    # --------------------------------------------------------
    #
    # Что делает:
    # Возвращает "ключ качества" решения, по которому мы сравниваем
    # два маршрута между собой.
    #
    # Качество определяется так:
    #   1. сначала максимизируем число посещённых мельниц;
    #   2. при равенстве максимизируем суммарный leftover.
    #
    # Почему именно так:
    # - главная цель - собрать как можно больше мельниц;
    # - leftover используется как tie-break:
    #   если две конфигурации собрали одинаковое число мельниц,
    #   предпочитаем ту, которая оставляет больше свободы по ходу.
    #
    # Это делает поиск чуть "аккуратнее":
    # алгоритм чаще предпочитает менее зажатые маршруты.
    def quality_key(self) -> Tuple[int, int]:
        return self.visited_count(), self.total_leftover()

    # --------------------------------------------------------
    # update_index_from
    # --------------------------------------------------------
    #
    # Что делает:
    # После вставки или удаления мельницы обновляет быстрые индексы
    # obj_route и obj_pos для всех мельниц маршрута, начиная с позиции from.
    #
    # Чтобы такие операции были быстрыми, мы храним:
    #   obj_route[obj] - в каком маршруте находится мельница,
    #   obj_pos[obj]   - на какой позиции она стоит.
    #
    # Но после изменения маршрута позиции некоторых мельниц сдвигаются.
    # Поэтому индексы надо синхронизировать.
    #
    # Почему обновляем не весь маршрут, а только хвост начиная с from:
    # всё, что стоит до from, не изменило своей позиции,
    # а значит пересчитывать это не нужно.
    def update_index_from(self, r: int, start_pos: int) -> None:
        if start_pos < 0:
            start_pos = 0
        route = self.routes[r]
        for pos in range(start_pos, len(route)):
            obj = route[pos]
            self.obj_route[obj] = r
            self.obj_pos[obj] = pos

    # --------------------------------------------------------
    # removal_delta_by_pos
    # --------------------------------------------------------
    #
    # Что делает:
    # Считает, на сколько уменьшится стоимость маршрута героя r,
    # если удалить мельницу из позиции pos.
    #
    # Что это даёт LNS:
    # Это симметричная идея к insertion_delta.
    # Вместо полного пересчёта маршрута мы смотрим только на локальное изменение.
    #
    # Зачем это нужно:
    # - destroy_worst использует эту величину, чтобы понять,
    #   какие мельницы "дорого" держать в маршруте;
    # - remove_object использует её, чтобы быстро обновить route_costs.
    #
    # Разбор случаев:
    #
    # 1. В маршруте одна мельница.
    #    Тогда после удаления маршрут становится пустым,
    #    и исчезает вся его стоимость.
    #
    # 2. Удаляем первую мельницу.
    #    Раньше маршрут начинался с x, а теперь начнётся с b.
    #    Значит:
    #      - убираем стартовую стоимость до x,
    #      + возвращаем стартовую стоимость до b,
    #      - убираем переход x -> b и VISIT_COST для x.
    #
    # 3. Удаляем последнюю мельницу.
    #    Просто убираем последний переход a -> x и VISIT_COST для x.
    #
    # 4. Удаляем мельницу из середины.
    #    Раньше был кусок a -> x -> b.
    #    После удаления он становится a -> b.
    #    Значит:
    #      - убираем dist(a, x),
    #      - убираем dist(x, b),
    #      + возвращаем dist(a, b),
    #      - убираем VISIT_COST для x.
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

    # --------------------------------------------------------
    # insertion_delta
    # --------------------------------------------------------
    #
    # Что делает:
    # Считает локальное изменение стоимости маршрута при вставке мельницы.
    #
    # "На сколько увеличится стоимость маршрута героя r,
    #  если вставить мельницу obj в позицию pos?"
    #
    #
    # Благодаря этому:
    # - greedy repair работает быстро;
    # - regret repair тоже остаётся приемлемым по времени.
    #
    # Разбираем случаи:
    #
    # 1. Маршрут пуст.
    #    Тогда стоимость вставки - это просто стартовая стоимость
    #    добраться до мельницы и посетить его.
    #
    # 2. Вставка в начало.
    #    Раньше первым была мельница b, теперь первым станет obj.
    #    Поэтому:
    #      + добавляем стоимость "старт -> obj"
    #      - убираем старую стоимость "старт -> b"
    #      + добавляем переход obj -> b и VISIT_COST для obj
    #
    # 3. Вставка в конец.
    #    Добавляем переход от последней мельницы к obj и VISIT_COST.
    #
    # 4. Вставка в середину.
    #    Раньше был кусок a -> b.
    #    После вставки он заменяется на a -> obj -> b.
    #    Поэтому:
    #      + dist(a, obj)
    #      + dist(obj, b)
    #      - dist(a, b)
    #      + VISIT_COST
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

    # --------------------------------------------------------
    # best_insertion_in_route
    # --------------------------------------------------------
    #
    # Эта функция ищет ЛУЧШЕЕ место вставки мельницы obj
    # в маршрут героя r.
    #
    # Что значит "лучшее":
    # - среди всех позиций 0..n выбираем ту,
    #   где insertion_delta минимальна.
    #
    # Но есть дополнительное ограничение:
    # - новый маршрут должен оставаться допустимым по ёмкости героя.
    #
    # Возвращаем:
    # - None, если вставить мельницу в этот маршрут нельзя;
    # - пару (best_delta, best_pos), если можно.
    #
    # Что это даёт LNS:
    # Repair-фаза LNS должна уметь отвечать на вопрос:
    # "Куда выгоднее всего вставить эту мельницу?"
    #
    # Эта функция - базовый кирпич для repair-операторов.
    # И greedy, и regret2 опираются именно на неё.
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

    # --------------------------------------------------------
    # insert
    # --------------------------------------------------------
    #
    # Реально вставляет мельницу в маршрут и обновляет все индексы.
    #
    # Что именно делает:
    # 1. Проверяет, что мельница ещё никому не назначена.
    # 2. Считает дельту стоимости (или берёт уже готовую, если она передана).
    # 3. Вставляет мельницу в вектор маршрута.
    # 4. Увеличивает кэшированную стоимость маршрута.
    # 5. Обновляет быстрые индексы obj_route / obj_pos.
    # 6. Увеличивает assigned_count.
    def insert(self, obj: int, r: int, pos: int, given_delta: Optional[int] = None) -> None:
        if self.assigned(obj):
            raise RuntimeError("insert: мельница уже назначена")

        delta = given_delta if given_delta is not None else self.insertion_delta(r, obj, pos)

        self.routes[r].insert(pos, obj)
        self.route_costs[r] += delta
        self.obj_route[obj] = r
        self.assigned_count += 1

        self.update_index_from(r, pos)

    # --------------------------------------------------------
    # remove_object
    # --------------------------------------------------------
    #
    # Реально удаляет мельницу из её маршрута.
    #
    # Что делает:
    # 1. По быстрым индексам находит маршрут и позицию мельницы.
    # 2. Считает, на сколько уменьшится стоимость маршрута.
    # 3. Удаляет мельницу из вектора маршрута.
    # 4. Уменьшает route_costs[r] на найденную дельту.
    # 5. Сбрасывает obj_route[obj] и obj_pos[obj].
    # 6. Обновляет индексы объектов, стоящих после удалённого.
    # 7. Уменьшает assigned_count.
    #
    # Возвращаемое значение:
    # - на сколько уменьшилась стоимость маршрута.
    #
    # Это и есть сердцевина large neighborhood search:
    # не мелкая перестановка, а удаление заметного куска решения
    # и последующая пересборка.
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

    # --------------------------------------------------------
    # validate_basic
    # --------------------------------------------------------
    #
    # Что проверяем:
    #
    # 1. route_costs[r] действительно совпадает с пересчётом route_cost(...).
    #    Это защищает нас от ошибок в дельтах вставки/удаления.
    #
    # 2. Стоимость маршрута не превышает hero_capacity + VISIT_COST.
    #    То есть решение удовлетворяет нашей упрощённой модели допустимости.
    #
    # 3. Каждая мельница:
    #    - имеет корректный индекс,
    #    - встречается не более одного раза,
    #    - согласована с таблицами obj_route и obj_pos.
    #
    # 4. assigned_count совпадает с реальным числом назначенных мельниц.
    def validate_basic(self) -> bool:
        seen = [0] * self.data.object_count
        cnt = 0

        for r in range(self.data.num_heroes):
            actual_cost = self.data.route_cost(r, self.routes[r])

            if actual_cost != self.route_costs[r]:
                return False
            if actual_cost > self.data.hero_capacity(r) + VISIT_COST:
                return False

            for pos, obj in enumerate(self.routes[r]):
                if obj < 0 or obj >= self.data.object_count:
                    return False
                if seen[obj]:
                    return False
                seen[obj] = 1

                if self.obj_route[obj] != r:
                    return False
                if self.obj_pos[obj] != pos:
                    return False

                cnt += 1

        return cnt == self.assigned_count


# ============================================================
# Операторы LNS
# ============================================================

# DestroyOp - оператор "разрушения" решения.
#
# На destroy-фазе мы намеренно убираем часть уже назначенных мельниц,
# чтобы затем repair-фаза попробовала собрать решение заново, но лучше.
class DestroyOp(Enum):
    # Случайное удаление:
    # хорошо добавляет разнообразие поиска.
    RANDOM = 0

    # Удаление "плохих" мельниц:
    # направляет поиск туда, где можно сильнее перестроить решение.
    WORST = 1


# RepairOp - оператор "восстановления" решения.
#
# На repair-фазе мы пытаемся снова вставить удалённые и ещё не назначенные мельницы.
class RepairOp(Enum):
    # Самый дешёвый следующий шаг.
    GREEDY = 0

    # Бережёт "хрупкие" мельницы, которые потом можно потерять.
    REGRET2 = 1


def destroy_op_to_string(op: DestroyOp) -> str:
    return "random" if op == DestroyOp.RANDOM else "worst"


def repair_op_to_string(op: RepairOp) -> str:
    return "greedy" if op == RepairOp.GREEDY else "regret2"


# ============================================================
# LNS-солвер одного дня
# ============================================================

@dataclass
class RunStats:
    iterations_done: int = 0
    accepted_moves: int = 0
    improving_moves: int = 0
    best_updates: int = 0


class LNSSolver:
    def __init__(self, data: DayData, cfg: Config, seed: int, day_time_limit: float):
        self.m_data = data
        self.m_cfg = cfg
        self.m_rng = random.Random(seed)
        self.m_day_time_limit = day_time_limit
        self.m_stats = RunStats()

    def better_key(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        if a[0] != b[0]:
            return a[0] > b[0]
        return a[1] > b[1]

    def randint(self, lo: int, hi: int) -> int:
        return self.m_rng.randint(lo, hi)

    def rand01(self) -> float:
        return self.m_rng.random()

    def compute_temperature(self, progress: float) -> float:
        if self.m_cfg.temp_start <= 0.0 or self.m_cfg.temp_end <= 0.0:
            return 0.0

        progress = min(1.0, max(0.0, progress))

        if abs(self.m_cfg.temp_start - self.m_cfg.temp_end) < 1e-15:
            return self.m_cfg.temp_start

        return self.m_cfg.temp_start * pow(self.m_cfg.temp_end / self.m_cfg.temp_start, progress)

    # --------------------------------------------------------
    # choose_q
    # --------------------------------------------------------
    #
    # Что делает:
    # Выбирает, сколько мельниц удалить на текущей destroy-фазе.
    #
    # Что это даёт LNS:
    # В Large Neighborhood Search важен размер разрушения.
    #
    # Если удалять слишком мало:
    # - поиск будет слишком "локальным";
    # - алгоритм может застрять рядом с текущим решением.
    #
    # Если удалять слишком много:
    # - repair будет почти собирать решение заново;
    # - потеряется польза от уже найденной хорошей структуры.
    #
    # Поэтому мы удаляем не фиксированное число мельниц,
    # а случайное число q в диапазоне:
    #   [destroy_frac_min * visited_count,
    #    destroy_frac_max * visited_count]
    #
    # Это даёт баланс:
    # - иногда разрушение мягкое,
    # - иногда более сильное,
    # что делает поиск разнообразнее.
    def choose_q(self, sol: Solution) -> int:
        if sol.visited_count() <= 0:
            return 0

        lo = max(1, int(math.floor(self.m_cfg.destroy_frac_min * sol.visited_count())))
        hi = max(lo, int(math.ceil(self.m_cfg.destroy_frac_max * sol.visited_count())))
        hi = min(hi, sol.visited_count())
        lo = min(lo, hi)

        return self.randint(lo, hi)

    # Если новое решение лучше - принимаем всегда.
    # Если хуже - иногда принимаем, особенно в начале поиска.
    def accept(self, cand: Solution, cur: Solution, temperature: float) -> bool:
        ck = cand.quality_key()
        uk = cur.quality_key()

        if ck == uk or self.better_key(ck, uk):
            return True

        delta = (
            (cand.visited_count() - cur.visited_count()) +
            (cand.total_leftover() - cur.total_leftover()) / 1_000_000.0
        )

        if temperature <= 0.0:
            return False

        prob = math.exp(delta / temperature)
        return self.rand01() < prob

    def build_initial_solution(self) -> Solution:
        # Начинаем с пустого решения
        # и жадно вставляем мельницы, пока можем.
        #
        # LNS не стартует "из ничего" - ему нужен хотя бы какой-то
        # корректный базовый маршрут, который потом можно перестраивать.
        sol = Solution.empty(self.m_data)
        self.repair_greedy(sol)

        log_msg(
            f"[day {self.m_data.day}] init  | "
            f"visited={sol.visited_count():<3} | leftover={sol.total_leftover():<6}"
        )

        return sol

    def destroy(self, sol: Solution, op: DestroyOp, q: int) -> None:
        if q <= 0 or sol.visited_count() == 0:
            return

        if op == DestroyOp.RANDOM:
            self.destroy_random(sol, q)
        else:
            self.destroy_worst(sol, q)

    # Случайное разрушение.
    #
    # Что это даёт LNS:
    # Иногда полезно ломать решение без сильной логики,
    # просто чтобы попробовать совершенно другую конфигурацию.
    def destroy_random(self, sol: Solution, q: int) -> None:
        assigned_objs = []
        for obj in range(self.m_data.object_count):
            if sol.assigned(obj):
                assigned_objs.append(obj)

        self.m_rng.shuffle(assigned_objs)

        q = min(q, len(assigned_objs))
        for i in range(q):
            sol.remove_object(assigned_objs[i])

    # --------------------------------------------------------
    # destroy_worst
    # --------------------------------------------------------
    #
    # Что делает:
    # Удаляет мельницы, удаление которых сильнее всего "облегчает" маршруты.
    #
    # Что это даёт LNS:
    # Такой оператор разрушает решение не случайно, а осмысленно.
    # Он чаще выбивает мельницы, которые выглядят дорогими или неудобными,
    # а значит repair может потом собрать маршруты более удачно.
    #
    # Почему не выбираем всегда самую плохую мельницу:
    # чтобы сохранить разнообразие поиска.
    #
    # Поэтому мы:
    # 1. сортируем кандидатов по выгоде удаления;
    # 2. берём верхние rcl_size кандидатов;
    # 3. случайно выбираем одного из них.
    def destroy_worst(self, sol: Solution, q: int) -> None:
        for _ in range(q):
            if sol.visited_count() == 0:
                break

            cands: List[Tuple[int, int]] = []
            for obj in range(self.m_data.object_count):
                if not sol.assigned(obj):
                    continue
                cands.append((sol.removal_delta(obj), obj))

            cands.sort(key=lambda x: (-x[0], x[1]))

            limit = min(len(cands), max(1, self.m_cfg.rcl_size))
            idx = self.randint(0, limit - 1)
            sol.remove_object(cands[idx][1])

    def repair(self, sol: Solution, op: RepairOp) -> None:
        if op == RepairOp.GREEDY:
            self.repair_greedy(sol)
        else:
            self.repair_regret2(sol)

    # --------------------------------------------------------
    # greedy_insert_one
    # --------------------------------------------------------
    #
    # Что делает:
    # Выбирает одну самую выгодную локальную вставку.
    #
    # Идея:
    # перебираем все ещё не назначенные мельницы и все маршруты,
    # для каждой мельницы ищем лучшую допустимую вставку,
    # а затем выбираем глобально самую дешёвую вставку.
    #
    # То есть на каждом шаге отвечаем на вопрос:
    # "Какую мельницу сейчас проще всего вставить в решение?"
    #
    # Это очень простой и понятный repair-оператор.
    #
    # Плюсы:
    # - легко понять;
    # - быстро даёт неплохие решения.
    #
    # Минусы:
    # - может "захватывать" лёгкие мельницы,
    #   оставляя сложные на потом, когда вставить их уже некуда.
    def greedy_insert_one(self, sol: Solution) -> bool:
        best_obj = -1
        best_r = -1
        best_pos = -1
        best_delta = 0
        found = False

        for obj in range(self.m_data.object_count):
            if sol.assigned(obj):
                continue

            for r in range(self.m_data.num_heroes):
                ins = sol.best_insertion_in_route(r, obj)
                if ins is None:
                    continue

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
            return False

        sol.insert(best_obj, best_r, best_pos, best_delta)
        return True

    def repair_greedy(self, sol: Solution) -> None:
        while self.greedy_insert_one(sol):
            pass

    # --------------------------------------------------------
    # regret2_insert_one
    # --------------------------------------------------------
    #
    # Что делает:
    # Выбирает не просто "самую дешёвую сейчас" вставку,
    # а мельницу, которую опасно откладывать на потом.
    #
    # Основная идея:
    # не все мельницы одинаково "срочные".
    #
    # Для одной мельницы может быть:
    # - один очень хороший вариант вставки,
    # - и второй вариант сильно хуже.
    #
    # Если такую мельницу не вставить сейчас,
    # позже хороший вариант может пропасть,
    # и мельницу станет трудно или невозможно вставить.
    #
    # Поэтому для каждой мельницы считаем:
    # - best1 = лучшая стоимость вставки;
    # - best2 = вторая лучшая стоимость вставки;
    # - regret = best2 - best1.
    #
    # Чем больше regret, тем важнее вставить мельницу прямо сейчас.
    #
    # На каждом шаге выбираем мельницу с максимальным regret.
    #
    # Плюсы по сравнению с greedy:
    # - чаще бережёт "хрупкие" мельницы,
    #   которые потом могут стать недоступны.
    #
    # Минусы:
    # - сложнее;
    # - обычно медленнее обычной жадной вставки.
    def regret2_insert_one(self, sol: Solution) -> bool:
        chosen_obj = -1
        chosen_r = -1
        chosen_pos = -1
        chosen_best_delta = 0
        best_regret = -10**18
        found = False

        BIG_M = 1_000_000

        for obj in range(self.m_data.object_count):
            if sol.assigned(obj):
                continue

            best1 = 10**18
            best2 = 10**18
            best_route = -1
            best_pos = -1

            for r in range(self.m_data.num_heroes):
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
            return False

        sol.insert(chosen_obj, chosen_r, chosen_pos, chosen_best_delta)
        return True

    def repair_regret2(self, sol: Solution) -> None:
        while self.regret2_insert_one(sol):
            pass

    # --------------------------------------------------------
    # solve
    # --------------------------------------------------------
    #
    # LNS = "разруши и почини" много раз подряд.
    #
    # current  - текущее решение, вокруг которого мы ищем.
    # best     - лучшее решение, найденное за всё время.
    #
    # На каждой итерации:
    # 1. создаём candidate = current;
    # 2. разрушаем candidate;
    # 3. чиним candidate;
    # 4. если candidate хороший - принимаем;
    # 5. если candidate лучший за всё время - запоминаем.
    #
    # Почему LNS вообще работает:
    #
    # - destroy ломает текущее решение и позволяет выйти из "ловушки";
    # - repair собирает его заново, потенциально в лучшей конфигурации;
    # - simulated annealing позволяет иногда принимать ухудшение,
    #   чтобы не застревать в локальном оптимуме слишком рано.
    def solve(self, deadline_sec: float) -> Tuple[Solution, RunStats]:
        if self.m_data.object_count == 0:
            return Solution.empty(self.m_data), self.m_stats

        # Строим стартовое решение.
        # Оно не обязано быть идеальным - достаточно, чтобы было корректным.
        current = self.build_initial_solution()
        best = current.clone()

        start = now_sec()

        while now_sec() < deadline_sec and (
            self.m_cfg.iterations <= 0 or self.m_stats.iterations_done < self.m_cfg.iterations
        ):
            self.m_stats.iterations_done += 1

            # progress от 0 до 1 - как далеко мы продвинулись по времени.
            elapsed = now_sec() - start
            progress = min(1.0, elapsed / max(1e-9, self.m_day_time_limit))

            # Температура падает со временем.
            temperature = self.compute_temperature(progress)

            # Просто случайно выбираем destroy и repair.
            d_op = DestroyOp.RANDOM if self.rand01() < 0.5 else DestroyOp.WORST
            r_op = RepairOp.GREEDY if self.rand01() < 0.5 else RepairOp.REGRET2

            # Работаем не с current напрямую, а с его копией.
            # Так проще сравнивать "до" и "после".
            cand = current.clone()

            q = self.choose_q(cand)
            self.destroy(cand, d_op, q)
            self.repair(cand, r_op)

            cur_key = current.quality_key()
            cand_key = cand.quality_key()

            # Решаем, принять ли новый кандидат.
            # Даже если он хуже, при ненулевой температуре
            # можем иногда принять его - это и есть simulated annealing.
            if self.accept(cand, current, temperature):
                if self.better_key(cand_key, cur_key):
                    self.m_stats.improving_moves += 1

                current = cand
                self.m_stats.accepted_moves += 1

                # Отдельно запоминаем глобально лучшее решение.
                if self.better_key(current.quality_key(), best.quality_key()):
                    best = current.clone()
                    self.m_stats.best_updates += 1

            if self.m_stats.iterations_done % self.m_cfg.log_every == 0:
                log_msg(
                    f"[day {self.m_data.day}] "
                    f"iter={self.m_stats.iterations_done:<6} | "
                    f"destroy={destroy_op_to_string(d_op):<7} | "
                    f"repair={repair_op_to_string(r_op):<7} | "
                    f"best=({best.visited_count():>3}, {best.total_leftover():>6}) | "
                    f"cur=({current.visited_count():>3}, {current.total_leftover():>6}) | "
                    f"temp={temperature:.6f}"
                )

        return best, self.m_stats


# ============================================================
# Результат за неделю
# ============================================================

@dataclass
class WeekResult:
    submission_by_hero: List[List[int]] = field(default_factory=list)
    total_visited: int = 0
    total_leftover: int = 0
    total_used_moves: int = 0


# ============================================================
# Сохранение результатов
# ============================================================

def save_submission_csv(
    path: Path,
    heroes_count: int,
    submission_by_hero: List[List[int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        fout.write("hero_id,object_id\n")

        for h in range(heroes_count):
            hero_id = h + 1
            for object_id in submission_by_hero[h]:
                fout.write(f"{hero_id},{object_id}\n")


def save_summary_txt(
    path: Path,
    total_visited: int,
    total_used_moves: int,
    total_leftover: int,
    fixed_hero_cost: int,
) -> None:
    reward = total_visited * 500
    net_score = reward - fixed_hero_cost

    with path.open("w", encoding="utf-8") as fout:
        fout.write(f"visited_total={total_visited}\n")
        fout.write(f"used_moves_total={total_used_moves}\n")
        fout.write(f"leftover_total={total_leftover}\n")
        fout.write(f"reward={reward}\n")
        fout.write(f"fixed_hero_cost={fixed_hero_cost}\n")
        fout.write(f"net_score={net_score}\n")


# ============================================================
# Решение всей недели
# ============================================================
#
# Между днями переносим:
# - где герой закончил день;
# - сколько движения условно "сохранилось".
def solve_week(full: FullData, cfg: Config) -> WeekResult:
    if cfg.heroes > full.hero_count():
        raise RuntimeError("Запрошено героев больше, чем доступно")

    result = WeekResult()
    result.submission_by_hero = [[] for _ in range(cfg.heroes)]

    hero_states: List[HeroState] = [HeroState(anchor_ext=0, carry_discount=0) for _ in range(cfg.heroes)]

    for day in range(1, DAYS + 1):
        day_limit = cfg.day_time_limits[day]

        log_msg("-" * 80)
        log_msg(f"DAY {day} | time_limit={int(day_limit)}")
        log_msg("-" * 80)

        day_data = DayData.build_for_day(full, day, cfg.heroes, hero_states)

        day_best = Solution.empty(day_data)
        day_stats = RunStats()

        if day_limit > 0.0 and day_data.object_count > 0:
            run_seed = cfg.seed + day * 10_000_019

            log_msg(
                f"[day {day}] seed={run_seed:<10} | mills={day_data.object_count:<4}"
            )

            solver = LNSSolver(day_data, cfg, run_seed, day_limit)
            day_best, day_stats = solver.solve(now_sec() + day_limit)

        if not day_best.validate_basic():
            raise RuntimeError(f"Решение дня {day} не прошло базовую проверку")

        # Переносим маршрут дня в общий недельный submission
        # и обновляем состояния героев на следующий день.
        for h in range(day_data.num_heroes):
            cap = day_data.hero_capacity(h)

            for obj in day_best.routes[h]:
                result.submission_by_hero[h].append(day_data.object_id(obj))

            if len(day_best.routes[h]) > 0:
                last_obj_internal = day_best.routes[h][-1]
                hero_states[h].anchor_ext = day_data.object_id(last_obj_internal)
                hero_states[h].carry_discount = max(0, cap - day_best.route_costs[h])
            else:
                hero_states[h].carry_discount += cap

        result.total_visited += day_best.visited_count()
        result.total_leftover += day_best.total_leftover()
        result.total_used_moves += day_best.total_used()

        log_msg(
            f"[day {day}] done   | "
            f"visited={day_best.visited_count():<3} | "
            f"leftover={day_best.total_leftover():<6} | "
            f"used_moves={day_best.total_used():<6} | "
            f"accepted={day_stats.accepted_moves:<6} | "
            f"improving={day_stats.improving_moves:<6} | "
            f"best_updates={day_stats.best_updates:<6}"
        )

    return result


# ============================================================
# main
# ============================================================

def main() -> int:
    try:
        cfg = parse_args()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        log_msg("=" * 80)
        log_msg("LNS SOLVER")
        log_msg("=" * 80)
        log_msg(f"Data dir: {cfg.data_dir}")
        log_msg(f"Output dir: {cfg.output_dir}")
        log_msg(f"Fixed heroes: {cfg.heroes}")
        log_msg("Init: greedy")
        log_msg("Destroy ops: random, worst")
        log_msg("Repair ops: greedy, regret2")

        full = FullData.load(cfg.data_dir)
        result = solve_week(full, cfg)

        fixed_hero_cost = cfg.heroes * HERO_COST
        reward = result.total_visited * 500
        net_score = reward - fixed_hero_cost

        submission_path = cfg.output_dir / "submission.csv"
        summary_path = cfg.output_dir / "summary.txt"

        save_submission_csv(submission_path, cfg.heroes, result.submission_by_hero)
        save_summary_txt(
            summary_path,
            result.total_visited,
            result.total_used_moves,
            result.total_leftover,
            fixed_hero_cost,
        )

        log_msg(f"CSV saved: {submission_path}")
        log_msg(f"Summary saved: {summary_path}")

        log_msg("=" * 80)
        log_msg("ИТОГИ")
        log_msg("=" * 80)
        log_msg(f"Visited total: {result.total_visited}")
        log_msg(f"Total used moves: {result.total_used_moves}")
        log_msg(f"Total leftover: {result.total_leftover}")
        log_msg(f"Reward: {reward}")
        log_msg(f"Fixed hero cost: {fixed_hero_cost}")
        log_msg(f"Net score: {net_score}")
        log_msg("=" * 80)

        return 0

    except Exception as e:
        print(f"{wall_timestamp()} | ERROR | {e}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
