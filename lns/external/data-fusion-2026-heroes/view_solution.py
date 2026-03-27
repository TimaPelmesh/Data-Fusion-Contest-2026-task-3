from __future__ import annotations

import argparse
import math
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import polars as pl

# На Windows включаем режим корректного DPI,
# чтобы окно не было "мыльным" на больших мониторах.
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import pygame

# ── Layout ───────────────────────────────────────────────────────────────────
INIT_WIDTH, INIT_HEIGHT = 1280, 800
DEFAULT_FPS = 30
MARGIN = 80
FRAME_T = 26
FIT_PADDING = FRAME_T + 55
GRID_STEP = 100

# Масштаб карты:
# - стартовый зум = fit * 0.9
# - минимальный зум = стартовый / 2
ZOOM_MIN_HARD = 0.01
ZOOM_MAX_HARD = 100.0

# Настройки "раздвигания" точек на карте,
# чтобы мельницы не слипались в один комок.
OVAL_RATIO = 1.92
DISPLAY_MIN_SEP = 34.0
DISPLAY_MIN_SEP_SOFT = 23.0
TAVERN_CLEARANCE = 58.0
SPREAD_ITERS = 28
SPREAD_ITERS_SOFT = 14
ANCHOR_STRENGTH = 0.10
MAX_FORCE_STEP = 7.0

# Цвета интерфейса
BG_COLOR = (247, 247, 244)
GRID_COLOR = (222, 220, 215)
STONE = (72, 60, 46)
STONE_LT = (110, 95, 75)
STONE_HI = (135, 118, 95)
STONE_DK = (45, 36, 26)
STONE_OUTER = (25, 20, 15)
GOLD = (170, 145, 48)
GOLD_DK = (125, 105, 32)
GEM_RED = (165, 28, 28)
GEM_HI = (225, 85, 85)
STATUS_BG = (30, 24, 18)
PARCHMENT = (210, 195, 165)
PANEL_BG = (245, 240, 228)
PANEL_BORDER = (130, 118, 95)
DARK_TEXT = (40, 32, 24)

TAV_WALL = (96, 62, 36)
TAV_WALL_LT = (138, 96, 62)
TAV_ROOF = (146, 48, 37)
TAV_ROOF_DK = (104, 33, 24)
TAV_DOOR = (30, 18, 10)
TAV_DOOR_HI = (92, 66, 38)
TAV_SIGN_BRD = (85, 65, 35)
TAV_SIGN_BG = (210, 185, 130)
TAV_WINDOW = (188, 219, 244)
TAV_WINDOW_F = (70, 52, 33)

BTN_BG = STONE
BTN_HOVER_BG = (95, 82, 65)
BTN_BORDER = GOLD_DK
BTN_TEXT = PARCHMENT

MEASURE_COLOR = (50, 40, 28)
MEASURE_LINE_COLOR = (90, 80, 65)
MEASURE_TEXT_BG = PANEL_BG
PIN_RING_COLOR = GOLD
UNVISITED_RED = (210, 45, 45)
OK_GREEN = (30, 150, 45)
INACTIVE_MILL_GRAY = (246, 246, 246)

DAY_COLORS = {
    1: (220, 40, 40),
    2: (240, 130, 20),
    3: (225, 205, 15),
    4: (40, 180, 60),
    5: (30, 160, 210),
    6: (50, 70, 195),
    7: (135, 40, 180),
}

# Классы "силы" героя по очкам хода.
# Это только для красивого отображения в интерфейсе.
HERO_CAPACITY_CLASSES = [
    (1450, "К1", (0, 122, 255)),
    (1600, "К2", (0, 180, 48)),
    (1750, "К3", (255, 145, 0)),
    (10**9, "К4", (180, 0, 255)),
]

FONT_SM = 18
FONT_MD = 23
FONT_TBL = 16
BLADE_SPIN_SPEED = 90.0
HERO_CELL_W = 72
HERO_CELL_H = 30
HERO_CELL_GAP = 4
HERO_ROWS = 16
HERO_ICON_SZ = 18

TBL_HDR_BG = (145, 133, 112)
TBL_HDR_TEXT = (250, 245, 235)
TBL_ROW1 = (242, 238, 228)
TBL_ROW2 = (233, 228, 218)
TBL_BORDER = (140, 128, 108)
TBL_PAD = 6
TBL_ROW_H = 24

MILL_STYLE_KIND = "tall_bold"
MILL_SIZE_SCALE = 0.90

MILL_SURFACE_CACHE = {}

# Игровые константы задачи
VISIT_COST = 100
HERO_COST = 2500


# ═══════════════ МИНИ-ДВИЖОК ДЛЯ ДАННЫХ И СИМУЛЯЦИИ ═══════════════════════════

@dataclass(frozen=True)
class Hero:
    """Простая запись про героя.

    frozen=True означает: после создания объект нельзя менять.
    Это удобно, потому что герой здесь — просто справочная информация.
    """

    hero_id: int
    move_points: int


@dataclass(frozen=True)
class GoldObject:
    """Простая запись про объект с золотом."""

    object_id: int
    day_open: int
    reward: int


@dataclass
class HeroState:
    """Текущее состояние героя в процессе симуляции.

    current_object:
        где герой сейчас находится
        0 означает стартовую таверну/замок

    current_day:
        текущий день недели

    current_move_points:
        сколько очков хода осталось прямо сейчас
    """

    current_object: int
    current_day: int
    current_move_points: int


@dataclass
class TransitionResult:
    """Результат одного перехода героя.

    То есть: герой был в точке A и поехал к объекту B.
    После симуляции мы знаем:
    - когда стартовал,
    - когда приехал,
    - когда покинул объект,
    - сколько потратил очков,
    - получил ли награду.
    """

    hero_id: int
    object_id_from: int
    object_id_to: int
    day_start: int
    day_arrive: int
    day_leave: int
    move_points_start: int
    move_points_arrive: int
    move_points_burned: int
    move_points_leave: int
    is_earlier: bool
    is_late: bool
    reward: int

    def as_dict(self) -> dict:
        """Преобразуем результат в обычный словарь.

        Зачем это нужно?
        Потому что потом удобно собрать список словарей и превратить его
        в polars DataFrame.
        """
        return {
            "hero_id": self.hero_id,
            "object_id_from": self.object_id_from,
            "object_id_to": self.object_id_to,
            "day_start": self.day_start,
            "day_arrive": self.day_arrive,
            "day_leave": self.day_leave,
            "move_points_start": self.move_points_start,
            "move_points_arrive": self.move_points_arrive,
            "move_points_burned": self.move_points_burned,
            "move_points_leave": self.move_points_leave,
            "is_earlier": self.is_earlier,
            "is_late": self.is_late,
            "reward": self.reward,
        }


class GameData:
    """Класс только для хранения игровых данных.

    Этот класс НЕ:
    - проверяет solution,
    - не считает score,
    - не симулирует маршруты.

    Он только:
    - читает входные csv-файлы,
    - хранит DataFrame-ы,
    - строит словари для быстрого доступа,
    - умеет отвечать на вопрос "какое расстояние от A до B?".
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)

        # Оставляем и DataFrame-ы, и словари.
        # DataFrame-ы нужны визуализатору,
        # а словари нужны для быстрой работы логики.
        self.heroes: pl.DataFrame = pl.DataFrame()
        self.objects: pl.DataFrame = pl.DataFrame()
        self.dist_start: pl.DataFrame = pl.DataFrame()
        self.dist_matrix: np.ndarray = np.zeros((0, 0), dtype=np.int32)

        self.hero_map: Dict[int, Hero] = {}
        self.object_map: Dict[int, GoldObject] = {}

        self.hero_mp_map: Dict[int, int] = {}
        self.object_day_map: Dict[int, int] = {}
        self.dist_start_map: Dict[int, int] = {}

        self._load()

    def _load(self):
        """Читаем все CSV и строим справочные структуры.

        Почему словари удобны:
        - hero_id -> move_points
        - object_id -> day_open
        - object_id -> dist_start

        Тогда не нужно каждый раз искать значение в таблице.
        """

        self.heroes = (
            pl.read_csv(self.data_dir / "data_heroes.csv")
            .select(
                pl.col("hero_id").cast(pl.Int32),
                pl.col("move_points").cast(pl.Int32),
            )
        )

        self.objects = (
            pl.read_csv(self.data_dir / "data_objects.csv")
            .select(
                pl.col("object_id").cast(pl.Int32),
                pl.col("day_open").cast(pl.Int32),
                pl.col("reward").cast(pl.Int32),
            )
        )

        self.dist_start = (
            pl.read_csv(self.data_dir / "dist_start.csv")
            .select(
                pl.col("object_id").cast(pl.Int32),
                pl.col("dist_start").cast(pl.Int32),
            )
        )

        # Матрица расстояний между объектами.
        # В CSV уже есть header, просто приводим всё к int.
        dist_objects = pl.read_csv(self.data_dir / "dist_objects.csv")
        self.dist_matrix = dist_objects.select(pl.all().cast(pl.Int32)).to_numpy()

        for row in self.heroes.iter_rows(named=True):
            hero = Hero(
                hero_id=int(row["hero_id"]),
                move_points=int(row["move_points"]),
            )
            self.hero_map[hero.hero_id] = hero
            self.hero_mp_map[hero.hero_id] = hero.move_points

        for row in self.objects.iter_rows(named=True):
            obj = GoldObject(
                object_id=int(row["object_id"]),
                day_open=int(row["day_open"]),
                reward=int(row["reward"]),
            )
            self.object_map[obj.object_id] = obj
            self.object_day_map[obj.object_id] = obj.day_open

        for row in self.dist_start.iter_rows(named=True):
            self.dist_start_map[int(row["object_id"])] = int(row["dist_start"])

    def get_distance(self, from_id: int, to_id: int) -> int:
        """Вернуть расстояние между двумя точками.

        Особый случай:
        from_id == 0 означает стартовую таверну/замок.
        Тогда берем расстояние из dist_start.csv.
        """
        if from_id == 0:
            return int(self.dist_start_map.get(to_id, 0))
        return int(self.dist_matrix[from_id - 1, to_id - 1])


class Solution:
    """Класс-обертка над solution CSV.

    Он умеет:
    - загрузить файл,
    - создать пустое решение,
    - сделать базовую очистку решения,
    - собрать маршруты по героям.

    Этот класс специально НЕ знает правил игры.
    Он работает только с таблицей.
    """

    def __init__(self, df: Optional[pl.DataFrame] = None):
        if df is None:
            df = self.empty_df()
        self.df = df

    @staticmethod
    def empty_df() -> pl.DataFrame:
        """Пустой DataFrame с правильными колонками."""
        return pl.DataFrame(
            {
                "hero_id": pl.Series([], dtype=pl.Int32),
                "object_id": pl.Series([], dtype=pl.Int32),
            }
        )

    @classmethod
    def empty(cls) -> "Solution":
        return cls(cls.empty_df())

    @classmethod
    def load(cls, path: Optional[Path]) -> "Solution":
        """Загрузить решение из CSV.

        Если путь не передан, возвращаем пустое решение.
        """
        if path is None:
            return cls.empty()
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        return cls(pl.read_csv(path))

    @property
    def height(self) -> int:
        return int(self.df.height)

    @property
    def is_empty(self) -> bool:
        return self.height == 0

    def basic_check(self, game: GameData) -> "Solution":
        """Базовая очистка решения по правилам соревнования.

        Что делаем:
        1. Проверяем, что есть колонки hero_id и object_id.
        2. Пытаемся привести их к int.
        3. Выкидываем строки с невалидными id.
        4. Удаляем повторные object_id, оставляя первое вхождение.
        5. Сохраняем исходный порядок строк.

        Последний пункт очень важен:
        порядок строк — это и есть порядок маршрута героя.
        """
        if self.is_empty:
            return Solution.empty()

        if "hero_id" not in self.df.columns or "object_id" not in self.df.columns:
            raise ValueError("В решении должны быть колонки 'hero_id' и 'object_id'")

        known_heroes = sorted(game.hero_mp_map.keys())
        known_objects = sorted(game.object_map.keys())

        clean_df = (
            self.df
            .with_row_index("row_id")
            .select(["row_id", "hero_id", "object_id"])
            .with_columns(
                pl.col("hero_id").cast(pl.Int32, strict=False),
                pl.col("object_id").cast(pl.Int32, strict=False),
            )
            .filter(
                pl.col("hero_id").is_not_null()
                & pl.col("object_id").is_not_null()
                & pl.col("hero_id").is_in(known_heroes)
                & pl.col("object_id").is_in(known_objects)
            )
            # Если один и тот же объект указан несколько раз,
            # оставляем только первое вхождение.
            .unique(subset=["object_id"], keep="first")
            .sort("row_id")
            .drop("row_id")
        )

        return Solution(clean_df)

    def routes_by_hero(self) -> Dict[int, list[int]]:
        """Собрать маршруты в виде:
        hero_id -> [object_id_1, object_id_2, ...]
        """
        routes: Dict[int, list[int]] = {}
        if self.is_empty:
            return routes

        for row in self.df.iter_rows(named=True):
            hero_id = int(row["hero_id"])
            object_id = int(row["object_id"])
            routes.setdefault(hero_id, []).append(object_id)

        return routes


class RouteSimulator:
    """Класс, который знает правила движения героев.

    Он берет:
    - GameData
    - Solution или маршрут одного героя

    И умеет:
    - симулировать переходы,
    - разворачивать решение в подробную таблицу,
    - считать итоговый score.
    """

    def __init__(self, game: GameData):
        self.game = game

    def simulate_transition(
        self,
        hero_id: int,
        current_state: HeroState,
        target_object: int,
    ) -> Optional[TransitionResult]:
        """Симулируем один переход героя к следующей мельнице.

        Здесь аккуратно реализованы игровые правила:
        - старт из таверны,
        - поездка к объекту,
        - перенос части дороги на следующий день,
        - ранний приезд и ожидание,
        - поздний приезд,
        - правило последнего шага.

        Важно:
        мы сохраняем логику, совместимую с официальным учебным utils.
        """

        hero = self.game.hero_map.get(hero_id)
        target = self.game.object_map.get(target_object)

        # На всякий случай проверяем, что id существуют.
        if hero is None or target is None:
            return None

        max_move_points = hero.move_points
        previous_object = current_state.current_object
        travel_distance = self.game.get_distance(previous_object, target_object)

        # Для первого объекта маршрута используем правило "герой сидит в таверне
        # до тех пор, пока не придет день первой цели".
        #
        # Это означает: стартуем не обязательно в день 1,
        # а в день открытия первой мельницы маршрута.
        #
        # Для учебных данных это допустимо и соответствует условию.
        if previous_object == 0:
            current_day = target.day_open
            current_move_points = max_move_points
        else:
            current_day = current_state.current_day
            current_move_points = current_state.current_move_points

        # Смотрим, хватает ли очков хода доехать за текущий день.
        diff_move_points = current_move_points - travel_distance

        if diff_move_points >= 0:
            # Хватило: приехали в тот же день.
            day_arrive = current_day
            move_points_arrive = diff_move_points
        else:
            # Не хватило: остаток дороги переносится на следующий день.
            #
            # В этих учебных данных такого одного переноса достаточно.
            # Именно такую логику использует официальный примерный utils.
            day_arrive = current_day + 1
            move_points_arrive = max_move_points + diff_move_points

        # Значения "по умолчанию".
        day_leave = day_arrive
        move_points_leave = 0
        move_points_burned = 0
        is_earlier = False
        is_late = False

        # Сравниваем день прибытия и день открытия мельницы.
        days_diff = day_arrive - target.day_open

        if days_diff < 0:
            # РАННИЙ ПРИЕЗД
            #
            # Герой пришел слишком рано и обязан ждать.
            # Остаток очков хода в день прибытия "сгорает".
            # Если ждать надо несколько дней, то сгорают и полные дневные запасы.
            is_earlier = True
            day_leave = target.day_open

            # Сколько полных дней он просто стоит между day_arrive и day_open.
            total_wasted_days = -days_diff - 1

            move_points_burned = move_points_arrive + max_move_points * total_wasted_days

            # В день открытия герой получает полный запас хода,
            # после чего платит 100 очков за посещение.
            move_points_leave = max_move_points - VISIT_COST

        elif days_diff == 0:
            # ПРИЕЗД ВОВРЕМЯ
            #
            # Если очков хватает на визит — просто списываем 100.
            # Если не хватает, но герой уже доехал до объекта,
            # работает правило "последнего шага": визит успешен, остаток = 0.
            if move_points_arrive >= VISIT_COST:
                move_points_leave = move_points_arrive - VISIT_COST
            else:
                move_points_leave = 0

        else:
            # ПОЗДНИЙ ПРИЕЗД
            #
            # Герой всё равно заходит на объект и тратит очки,
            # но золото уже не получает.
            is_late = True
            if move_points_arrive >= VISIT_COST:
                move_points_leave = move_points_arrive - VISIT_COST
            else:
                move_points_leave = 0

        # Награда есть только если не опоздали.
        reward = 0 if is_late else target.reward

        return TransitionResult(
            hero_id=hero_id,
            object_id_from=previous_object,
            object_id_to=target_object,
            day_start=current_day,
            day_arrive=day_arrive,
            day_leave=day_leave,
            move_points_start=current_move_points,
            move_points_arrive=move_points_arrive,
            move_points_burned=move_points_burned,
            move_points_leave=move_points_leave,
            is_earlier=is_earlier,
            is_late=is_late,
            reward=reward,
        )

    def hero_journey(self, hero_id: int, object_ids: list[int]) -> list[dict]:
        """Симулировать весь маршрут одного героя."""
        current_state = HeroState(
            current_object=0,
            current_day=1,
            current_move_points=self.game.hero_mp_map.get(hero_id, 0),
        )

        journey_rows: list[dict] = []

        for target_object in object_ids:
            transition = self.simulate_transition(hero_id, current_state, target_object)
            if transition is None:
                continue

            journey_rows.append(transition.as_dict())

            # Обновляем состояние героя после посещения цели.
            current_state = HeroState(
                current_object=transition.object_id_to,
                current_day=transition.day_leave,
                current_move_points=transition.move_points_leave,
            )

        return journey_rows

    def expand_solution(
        self,
        submit: Solution | pl.DataFrame,
        remove_out_of_time: bool = False,
    ) -> pl.DataFrame:
        """Развернуть короткое решение в подробную таблицу переходов.

        Было:
            hero_id, object_id

        Станет:
            hero_id, object_id_from, object_id_to, day_arrive, reward, ...

        Это очень удобно для:
        - визуализации,
        - отладки,
        - таблиц-подсказок.
        """
        if isinstance(submit, Solution):
            df = submit.df
        else:
            df = submit

        if df.height == 0:
            return pl.DataFrame()

        # Очень важно не потерять порядок маршрута.
        # Поэтому не используем group_by, а собираем маршруты вручную.
        routes: Dict[int, list[int]] = {}
        for row in df.iter_rows(named=True):
            hero_id = int(row["hero_id"])
            object_id = int(row["object_id"])
            routes.setdefault(hero_id, []).append(object_id)

        expanded_rows: list[dict] = []
        for hero_id in sorted(routes.keys()):
            expanded_rows.extend(self.hero_journey(hero_id, routes[hero_id]))

        if not expanded_rows:
            return pl.DataFrame()

        expanded_df = pl.DataFrame(expanded_rows)

        wanted_cols = [
            "hero_id",
            "object_id_from",
            "object_id_to",
            "day_start",
            "day_arrive",
            "day_leave",
            "move_points_start",
            "move_points_arrive",
            "move_points_burned",
            "move_points_leave",
            "is_earlier",
            "is_late",
            "reward",
        ]
        expanded_df = expanded_df.select(wanted_cols)

        if remove_out_of_time:
            expanded_df = expanded_df.filter(pl.col("day_arrive") <= 7)

        return expanded_df

    def evaluate_solution(self, submit: Solution | pl.DataFrame) -> int:
        """Посчитать итоговый score по правилам соревнования."""
        if isinstance(submit, Solution):
            raw_solution = submit
        else:
            raw_solution = Solution(submit)

        checked = raw_solution.basic_check(self.game)
        if checked.is_empty:
            return 0

        detailed = self.expand_solution(checked)
        if detailed.height == 0:
            return 0

        total_reward = int(detailed["reward"].sum())
        max_hero_id = int(detailed["hero_id"].max())

        return int(total_reward - max_hero_id * HERO_COST)


# ═══════════════ UI-ЭЛЕМЕНТЫ И МЕЛКИЕ СТРУКТУРЫ ═════════════════════════════
@dataclass
class VisitInfo:
    hero_id: int
    visit_day: int
    open_day: int
    reward: int
    on_time: bool
    is_earlier: bool
    is_late: bool


class UIButton:
    """Очень простая кнопка.

    Она умеет:
    - хранить свой прямоугольник,
    - понимать, наведен ли курсор,
    - рисовать себя.
    """

    def __init__(
        self,
        text,
        w=143,
        h=33,
        bg=BTN_BG,
        hover_bg=BTN_HOVER_BG,
        border=BTN_BORDER,
        text_color=BTN_TEXT,
    ):
        self.text = text
        self.rect = pygame.Rect(0, 0, w, h)
        self._h = False
        self.bg = bg
        self.hover_bg = hover_bg
        self.border = border
        self.text_color = text_color

    def set_position(self, x, y):
        self.rect.topleft = (x, y)

    def set_size(self, w, h=None):
        self.rect.width = int(w)
        if h is not None:
            self.rect.height = int(h)

    def handle_motion(self, p):
        self._h = self.rect.collidepoint(p)

    def is_clicked(self, p):
        return self.rect.collidepoint(p)

    def draw(self, surf, font):
        pygame.draw.rect(
            surf,
            self.hover_bg if self._h else self.bg,
            self.rect,
            border_radius=4,
        )
        pygame.draw.rect(surf, self.border, self.rect, 1, border_radius=4)
        l = font.render(self.text, True, self.text_color)
        surf.blit(l, l.get_rect(center=self.rect.center))


# ═══════════════ DATA / GEOMETRY HELPERS ═════════════════════════════════════
def validate_data_dir(data_dir: Path):
    """Проверяем, что в папке есть все нужные CSV."""
    required = [
        "data_heroes.csv",
        "data_objects.csv",
        "dist_start.csv",
        "dist_objects.csv",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"В директории {data_dir} отсутствуют файлы: {', '.join(missing)}"
        )


def ensure_finite_xy(xy):
    """Аккуратно приводим координаты к массиву shape=(n, 2).

    Идея очень простая:
    - иногда данные бывают странного формата,
    - иногда там могут быть NaN/inf,
    - для рисования Pygame нужны нормальные конечные числа.

    Поэтому здесь всё "чистим" и приводим к безопасному виду.
    """
    arr = np.asarray(xy, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            arr = arr.reshape(1, 2)
        else:
            arr = arr.reshape(-1, 2)
    if arr.shape[1] < 2:
        pad = np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=float)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    arr = np.array(arr, dtype=float, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def classical_mds_layout(D: np.ndarray) -> Optional[np.ndarray]:
    """Построить 2D-координаты по матрице расстояний через classical MDS.

    Объяснение "по-школьному":
    у нас есть только расстояния между точками,
    а самих координат на плоскости нет.
    MDS пытается расставить точки так,
    чтобы расстояния на картинке были похожи на реальные расстояния из задачи.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] == 0:
        return None

    D = np.maximum(D, 0.0)
    D = 0.5 * (D + D.T)

    n = D.shape[0]
    D2 = D * D
    J = np.eye(n) - np.ones((n, n), dtype=float) / n
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)

    try:
        evals, evecs = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        return None

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    pos = evals > 1e-9
    if not np.any(pos):
        return None

    dims = min(2, int(pos.sum()))
    vals = np.clip(evals[:dims], 0.0, None)
    vecs = evecs[:, :dims]
    X = vecs * np.sqrt(vals)[None, :]

    if dims < 2:
        X = np.hstack([X, np.zeros((n, 1), dtype=float)])

    if not np.isfinite(X).all():
        return None

    return X[:, :2]


def spread_layout(
    points: np.ndarray,
    anchors: np.ndarray,
    min_sep: float,
    tavern_clearance: float,
    iters: int,
    anchor_strength: float = ANCHOR_STRENGTH,
    max_step: float = MAX_FORCE_STEP,
):
    """Немного "раздвигаем" точки, чтобы они не налезали друг на друга.

    Здесь идея как у маленькой физической симуляции:
    - если две точки слишком близко, они отталкиваются;
    - если точка слишком залезла в центр (на таверну), её выталкиваем наружу;
    - при этом слегка тянем точки обратно к их исходным позициям,
      чтобы карта не развалилась совсем.
    """
    pts = ensure_finite_xy(points).copy()
    anc = ensure_finite_xy(anchors).copy()
    n = len(pts)
    if n == 0:
        return pts

    min_sep = max(1e-6, float(min_sep))
    tavern_clearance = max(0.0, float(tavern_clearance))
    max_step = max(1e-6, float(max_step))

    golden = 2.399963229728653

    for _ in range(max(0, int(iters))):
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)

        mask = dist < min_sep
        safe = np.where(mask, np.maximum(dist, 1e-6), 1.0)
        force_mag = np.where(mask, (min_sep - dist) / safe * 0.5, 0.0)
        disp = np.sum(delta * force_mag[..., None], axis=1)

        rad = np.linalg.norm(pts, axis=1)
        inside = rad < tavern_clearance
        if np.any(inside):
            dirs = pts.copy()
            zero = rad < 1e-6
            if np.any(zero):
                ids = np.arange(n)[zero]
                ang = (ids + 1.0) * golden
                dirs[zero, 0] = np.cos(ang)
                dirs[zero, 1] = np.sin(ang)
            dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)
            push = (tavern_clearance - rad.clip(min=0.0))[:, None] * 0.7
            disp[inside] += dirs[inside] * push[inside]

        step = np.linalg.norm(disp, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(step, 1e-6))
        pts += disp * scale[:, None]
        pts += float(anchor_strength) * (anc - pts)

    return pts


def build_layout(inst: GameData):
    """Построить 2D-координаты для всех объектов и таверны.

    Что происходит:
    1. Собираем полную матрицу расстояний, включая таверну как узел 0.
    2. Через MDS получаем примерные координаты.
    3. Немного поворачиваем карту для красоты.
    4. Растягиваем по X, чтобы картинка была "овальной".
    5. Раздвигаем точки, чтобы они не слипались.
    """
    object_ids = sorted(int(x) for x in inst.objects["object_id"].to_list())
    n = len(object_ids)

    obj_dm = np.asarray(inst.dist_matrix, dtype=float)
    obj_dm = 0.5 * (obj_dm + obj_dm.T)

    d0 = np.array([float(inst.dist_start_map[int(oid)]) for oid in object_ids], dtype=float)

    full_D = np.zeros((n + 1, n + 1), dtype=float)
    full_D[1:, 1:] = obj_dm
    full_D[0, 1:] = d0
    full_D[1:, 0] = d0

    coords = classical_mds_layout(full_D)
    if coords is None:
        # Если MDS не получилось, строим "аварийную" раскладку:
        # размещаем точки по кругу.
        coords = np.zeros((n + 1, 2), dtype=float)
        ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        coords[1:, 0] = d0 * np.cos(ang)
        coords[1:, 1] = d0 * np.sin(ang)

    coords = ensure_finite_xy(coords)
    coords -= coords[0]
    coords[0] = np.array([0.0, 0.0], dtype=float)

    # Поворачиваем карту так, чтобы она лучше смотрелась.
    if n > 1:
        obj = coords[1:]
        obj_center = obj - obj.mean(axis=0, keepdims=True)
        cov = obj_center.T @ obj_center
        try:
            evals, evecs = np.linalg.eigh(cov)
            pc = evecs[:, int(np.argmax(evals))]
            ang = math.atan2(float(pc[1]), float(pc[0]))
            c, s = math.cos(-ang), math.sin(-ang)
            R = np.array([[c, -s], [s, c]], dtype=float)
            coords = coords @ R.T
        except Exception:
            pass

    coords[:, 0] *= OVAL_RATIO

    if n > 0:
        obj = coords[1:].copy()
        anchors = obj.copy()

        bb_min = obj.min(axis=0)
        bb_max = obj.max(axis=0)
        diag = float(np.linalg.norm(bb_max - bb_min))
        if not np.isfinite(diag) or diag <= 0:
            diag = 1000.0

        hard_sep = max(diag * 0.022, 18.0)
        soft_sep = max(diag * 0.015, 12.0)
        tav_clear = max(diag * 0.055, 45.0)
        hard_step = max(diag * 0.006, 4.0)
        soft_step = max(diag * 0.0045, 3.0)

        obj = spread_layout(
            obj,
            anchors,
            min_sep=hard_sep,
            tavern_clearance=tav_clear,
            iters=SPREAD_ITERS,
            anchor_strength=ANCHOR_STRENGTH,
            max_step=hard_step,
        )
        obj = spread_layout(
            obj,
            anchors,
            min_sep=soft_sep,
            tavern_clearance=tav_clear * 0.92,
            iters=SPREAD_ITERS_SOFT,
            anchor_strength=ANCHOR_STRENGTH * 0.8,
            max_step=soft_step,
        )
        coords[1:] = obj
        coords[0] = np.array([0.0, 0.0], dtype=float)

    node_to_idx = {0: 0}
    for i, oid in enumerate(object_ids, start=1):
        node_to_idx[int(oid)] = i

    return coords, object_ids, node_to_idx


# ═══════════════ COLOR HELPERS ═══════════════════════════════════════════════
def _mix_color(c1, c2, t: float):
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(c1[i] * (1.0 - t) + c2[i] * t) for i in range(3))


def blend_colors(base, tint, alpha: float):
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(int(base[i] * (1.0 - alpha) + tint[i] * alpha) for i in range(3))


# ═══════════════ ICONS ═══════════════════════════════════════════════════════
def _tavern_geometry(cx, cy, size):
    cx, cy, s = float(cx), float(cy), max(10.0, float(size))

    half_w = max(8.0, s * 0.82)
    wall_h = max(8.0, s * 0.72)
    roof_h = max(8.0, s * 0.55)

    right = cx + half_w
    top = cy - wall_h / 2.0

    roof_peak = (cx, top - roof_h)
    roof_right = (right + 6.0, top + 2.0)

    ch_w = max(5.0, s * 0.14)
    ch_h = max(10.0, s * 0.38)
    ch_x = cx + half_w * 0.28

    denom = max(roof_right[0] - roof_peak[0], 1.0)
    t = (ch_x - roof_peak[0]) / denom
    t = max(0.0, min(1.0, t))
    roof_y = roof_peak[1] + t * (roof_right[1] - roof_peak[1])

    ch_rect = (
        ch_x - ch_w / 2.0,
        roof_y - ch_h + 5.0,
        ch_w,
        ch_h,
    )
    smoke_x = ch_x
    smoke_y = ch_rect[1] - 1.0
    return smoke_x, smoke_y


def draw_tavern_icon(surface, cx, cy, size, _c=None):
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(size)):
        return

    cx, cy, s = int(cx), int(cy), max(10, int(size))

    half_w = max(8, int(s * 0.82))
    wall_h = max(8, int(s * 0.72))
    roof_h = max(8, int(s * 0.55))

    left = cx - half_w
    right = cx + half_w
    top = cy - wall_h // 2
    bot = cy + wall_h // 2

    roof_left = (left - 6, top + 2)
    roof_right = (right + 6, top + 2)
    roof_peak = (cx, top - roof_h)

    ch_w = max(5, int(s * 0.14))
    ch_h = max(10, int(s * 0.38))
    ch_x = cx + int(half_w * 0.28)

    denom = max(roof_right[0] - roof_peak[0], 1)
    t = (ch_x - roof_peak[0]) / denom
    t = max(0.0, min(1.0, t))
    roof_y = roof_peak[1] + t * (roof_right[1] - roof_peak[1])

    ch_rect = pygame.Rect(
        ch_x - ch_w // 2,
        int(roof_y - ch_h + 5),
        ch_w,
        ch_h,
    )

    pygame.draw.rect(surface, STONE, ch_rect, border_radius=2)
    pygame.draw.rect(surface, STONE_DK, (ch_rect.x - 1, ch_rect.y - 2, ch_rect.w + 2, 3), border_radius=1)
    pygame.draw.line(surface, STONE_LT, (ch_rect.left + 1, ch_rect.top + 1), (ch_rect.left + 1, ch_rect.bottom - 2), 1)

    pygame.draw.polygon(surface, TAV_ROOF, [roof_left, roof_right, roof_peak])
    pygame.draw.line(surface, TAV_ROOF_DK, roof_left, roof_right, max(1, s // 10))

    wall_rect = pygame.Rect(left, top, 2 * half_w, wall_h)
    pygame.draw.rect(surface, TAV_WALL, wall_rect, border_radius=3)
    pygame.draw.rect(surface, TAV_WALL_LT, (left, top, max(6, half_w // 2), wall_h), border_radius=3)

    step = max(4, wall_h // 4)
    for yy in range(top + step, bot, step):
        pygame.draw.line(surface, TAV_DOOR_HI, (left + 2, yy), (right - 2, yy), 1)

    door_w = max(7, int(s * 0.34))
    door_h = max(10, int(s * 0.46))
    door_rect = pygame.Rect(cx - door_w // 2, bot - door_h, door_w, door_h)
    door_frame = door_rect.inflate(4, 3)

    pygame.draw.rect(surface, (150, 114, 68), door_frame, border_radius=3)
    pygame.draw.rect(surface, (18, 10, 5), door_rect, border_radius=3)
    pygame.draw.rect(surface, (176, 138, 84), door_rect, 1, border_radius=3)
    pygame.draw.circle(surface, TAV_DOOR_HI, (door_rect.right - 4, door_rect.centery), 2)

    win_sz = max(6, int(s * 0.22))
    win_y = top + max(5, int(s * 0.18))
    for wx in (cx - half_w // 2, cx + half_w // 2):
        r = pygame.Rect(wx - win_sz // 2, win_y, win_sz, win_sz)
        pygame.draw.rect(surface, TAV_WINDOW, r, border_radius=2)
        pygame.draw.rect(surface, TAV_WINDOW_F, r, 1, border_radius=2)
        pygame.draw.line(surface, TAV_WINDOW_F, (r.centerx, r.top), (r.centerx, r.bottom), 1)
        pygame.draw.line(surface, TAV_WINDOW_F, (r.left, r.centery), (r.right, r.centery), 1)

    attic = pygame.Rect(
        cx - max(4, win_sz // 2),
        top - roof_h // 2 - max(2, win_sz // 4),
        max(8, win_sz),
        max(6, int(win_sz * 0.8)),
    )
    pygame.draw.rect(surface, TAV_WINDOW, attic, border_radius=2)
    pygame.draw.rect(surface, TAV_WINDOW_F, attic, 1, border_radius=2)

    sx = right
    sy = top + max(4, int(s * 0.18))
    arm = max(8, int(s * 0.35))
    pygame.draw.line(surface, TAV_SIGN_BRD, (sx, sy), (sx + arm, sy), max(1, s // 12))
    sign = pygame.Rect(
        sx + arm - max(7, int(s * 0.14)),
        sy + 1,
        max(12, int(s * 0.26)),
        max(8, int(s * 0.18)),
    )
    pygame.draw.rect(surface, TAV_SIGN_BG, sign, border_radius=2)
    pygame.draw.rect(surface, TAV_SIGN_BRD, sign, 1, border_radius=2)


def draw_alpha_circle(surface, color_rgba, center, radius):
    r = max(1, int(radius))
    tmp = pygame.Surface((2 * r + 8, 2 * r + 8), pygame.SRCALPHA)
    pygame.draw.circle(tmp, color_rgba, (r + 4, r + 4), r)
    surface.blit(tmp, (int(center[0]) - r - 4, int(center[1]) - r - 4))


def draw_tavern_smoke(surface, cx, cy, size, t):
    sx, sy = _tavern_geometry(cx, cy, size)
    base = np.array([sx, sy], dtype=float)

    for i in range(8):
        phase = (t * 0.48 + i * 0.13) % 1.0
        wobble = math.sin(phase * math.pi * 2.4 + i * 0.85)
        drift_x = wobble * (size * 0.13) + i * 0.35
        rise = phase * size * 1.75
        puff_x = base[0] + drift_x
        puff_y = base[1] - rise
        rad = size * (0.12 + 0.10 * phase + i * 0.004)

        alpha_main = int(max(0, 165 * (1.0 - phase)))
        alpha_core = int(max(0, 105 * (1.0 - phase)))
        light = int(224 + 22 * (1.0 - phase))
        gray = int(188 + 18 * (1.0 - phase))

        draw_alpha_circle(surface, (light, light, light, alpha_main), (puff_x, puff_y), rad)
        draw_alpha_circle(surface, (gray, gray, gray, alpha_core), (puff_x - rad * 0.08, puff_y + rad * 0.03), rad * 0.58)


def draw_hero_figure(surface, cx, cy, size, color):
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(size)):
        return

    cx, cy, s = int(cx), int(cy), max(6, int(size))
    shadow = _mix_color(color, STONE_OUTER, 0.35)

    head_r = max(3, int(s * 0.22))
    head_y = cy - int(s * 0.32)
    pygame.draw.circle(surface, color, (cx, head_y), head_r)

    shoulder_y = head_y + head_r + 1
    waist_y = cy + int(s * 0.04)
    hem_y = cy + int(s * 0.19)

    shoulder_w = max(3, int(s * 0.28))
    waist_w = max(3, int(s * 0.20))
    hem_w = max(4, int(s * 0.34))

    cape = [
        (cx - hem_w - 1, shoulder_y + 1),
        (cx + hem_w + 1, shoulder_y + 1),
        (cx + hem_w - 1, hem_y + 1),
        (cx - hem_w + 1, hem_y + 1),
    ]
    pygame.draw.polygon(surface, shadow, cape)

    torso = [
        (cx - shoulder_w, shoulder_y),
        (cx + shoulder_w, shoulder_y),
        (cx + waist_w, waist_y),
        (cx + hem_w, hem_y),
        (cx - hem_w, hem_y),
        (cx - waist_w, waist_y),
    ]
    pygame.draw.polygon(surface, color, torso)

    arm_y = shoulder_y + int(s * 0.10)
    arm_t = max(3, int(s * 0.16))
    left_arm_end = (cx - hem_w - 1, arm_y + int(s * 0.04))
    right_arm_end = (cx + hem_w + int(s * 0.12), arm_y - int(s * 0.10))
    pygame.draw.line(surface, color, (cx - shoulder_w + 1, arm_y), left_arm_end, arm_t)
    pygame.draw.line(surface, color, (cx + shoulder_w - 1, arm_y), right_arm_end, arm_t)

    leg_y = cy + int(s * 0.52)
    leg_dx = max(2, int(s * 0.14))
    leg_t = max(3, int(s * 0.15))
    pygame.draw.line(surface, color, (cx - leg_dx, hem_y), (cx - leg_dx - 1, leg_y), leg_t)
    pygame.draw.line(surface, color, (cx + leg_dx, hem_y), (cx + leg_dx + 1, leg_y), leg_t)

    shield_w = max(3, int(s * 0.18))
    shield_h = max(5, int(s * 0.28))
    shield_x = cx - hem_w - shield_w // 2 - 2
    shield_y = arm_y + max(1, int(s * 0.02))
    pygame.draw.ellipse(surface, color, (shield_x, shield_y - shield_h // 2, shield_w, shield_h))

    sword_t = max(2, int(s * 0.11))
    sword_start = (cx + hem_w + 1, arm_y - 1)
    sword_end = (cx + hem_w + int(s * 0.26), arm_y - int(s * 0.26))
    pygame.draw.line(surface, color, sword_start, sword_end, sword_t)
    pygame.draw.line(
        surface,
        color,
        (sword_start[0] - int(s * 0.05), sword_start[1] + int(s * 0.03)),
        (sword_start[0] + int(s * 0.08), sword_start[1] - int(s * 0.03)),
        max(2, sword_t),
    )


# ═══════════════ MILL STYLE ══════════════════════════════════════════════════
def _mill_style_cfg():
    return {
        "body_w": 0.40,
        "body_h": 0.50,
        "body_y": 0.09,
        "top_ratio": 0.52,
        "sail_len": 0.55,
        "sail_root": 0.040,
        "sail_tip": 0.120,
        "body_fill_mix": 0.34,
        "roof_mix": 0.06,
        "sail_fill_mix": 0.72,
        "edge_mix": 0.52,
        "body_border_w": 2,
        "sail_border_w": 2,
        "hub_r": 0.065,
        "door": True,
    }


def _draw_panel_sails(surface, hx, hy, length, fill_color, edge_color, angle_offset=0.0, root_w=3, tip_w=7, edge_width=1):
    angs = [45, 135, 225, 315]
    for a in angs:
        r = math.radians(a + angle_offset)
        ex = math.cos(r)
        ey = -math.sin(r)
        nx, ny = -ey, ex

        root = max(2, root_w)
        tip = max(root + 1, tip_w)

        p1 = (hx + nx * root, hy + ny * root)
        p2 = (hx - nx * root, hy - ny * root)
        p3 = (hx + ex * length - nx * tip * 0.35, hy + ey * length - ny * tip * 0.35)
        p4 = (hx + ex * length + nx * tip * 0.75, hy + ey * length + ny * tip * 0.75)

        pts = [(int(x), int(y)) for x, y in [p1, p2, p3, p4]]
        pygame.draw.polygon(surface, fill_color, pts)
        pygame.draw.polygon(surface, edge_color, pts, edge_width)


def draw_mill_icon(surface, cx, cy, size, day_color, angle_offset=0.0, style=MILL_STYLE_KIND):
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(size)):
        return

    cx, cy, s = int(cx), int(cy), max(5, int(size))
    cfg = _mill_style_cfg()

    warm_dark = (88, 68, 54)
    warm_fill = (176, 150, 122)
    roof_base = (132, 98, 72)

    edge = _mix_color(day_color, warm_dark, cfg["edge_mix"])
    body_fill = _mix_color(day_color, warm_fill, cfg["body_fill_mix"])
    roof_fill = _mix_color(day_color, roof_base, cfg["roof_mix"])
    sail_fill = _mix_color(day_color, (252, 251, 244), cfg["sail_fill_mix"])

    bwb = max(4, int(s * cfg["body_w"]))
    bhh = max(4, int(s * cfg["body_h"]))
    bcy = cy + int(s * cfg["body_y"])
    bwt = max(2, int(bwb * cfg["top_ratio"]))

    bt = bcy - bhh
    bb = bcy + bhh
    hx, hy = cx, bt

    _draw_panel_sails(
        surface,
        hx,
        hy,
        max(5, int(s * cfg["sail_len"])),
        fill_color=sail_fill,
        edge_color=edge,
        angle_offset=angle_offset,
        root_w=max(2, int(s * cfg["sail_root"])),
        tip_w=max(4, int(s * cfg["sail_tip"])),
        edge_width=cfg["sail_border_w"],
    )

    body_pts = [(cx - bwb, bb), (cx + bwb, bb), (cx + bwt, bt), (cx - bwt, bt)]
    pygame.draw.polygon(surface, body_fill, body_pts)
    pygame.draw.polygon(surface, edge, body_pts, cfg["body_border_w"])

    roof_h = max(3, int(s * 0.12))
    roof_pts = [(cx - bwt - 2, bt + 1), (cx + bwt + 2, bt + 1), (cx, bt - roof_h)]
    pygame.draw.polygon(surface, roof_fill, roof_pts)
    pygame.draw.polygon(surface, edge, roof_pts, 1)

    highlight = _mix_color(body_fill, (255, 255, 255), 0.18)
    side_poly = [(cx - bwb + 1, bb - 1), (cx - max(2, bwb // 4), bb - 2), (cx - max(1, bwt // 4), bt + 2), (cx - bwt + 1, bt + 2)]
    pygame.draw.polygon(surface, highlight, side_poly)

    if cfg["door"]:
        dw = max(3, int(s * 0.12))
        dh = max(5, int(s * 0.20))
        d = pygame.Rect(cx - dw // 2, bb - dh, dw, dh)
        pygame.draw.rect(surface, _mix_color(body_fill, edge, 0.58), d, border_radius=2)
        pygame.draw.rect(surface, edge, d, 1, border_radius=2)

    hub_r = max(2, int(s * cfg["hub_r"]))
    pygame.draw.circle(surface, edge, (hx, hy), hub_r)
    pygame.draw.circle(surface, _mix_color(day_color, (255, 255, 255), 0.28), (hx, hy), max(1, hub_r - 2))

    pygame.draw.line(surface, edge, (cx - bwb + 1, bb), (cx + bwb - 1, bb), max(1, cfg["body_border_w"]))


def _mill_cache_key(size, color, angle):
    size_k = int(max(1, round(size)))
    color_k = tuple(int(v) for v in color)
    angle_k = (int(round(angle / 6.0)) * 6) % 360
    return size_k, color_k, angle_k


def get_cached_mill_surface(size, color, angle):
    key = _mill_cache_key(size, color, angle)
    if key in MILL_SURFACE_CACHE:
        return MILL_SURFACE_CACHE[key]

    size_k, color_k, angle_k = key
    side = max(24, int(math.ceil(size_k * 3.0)))
    surf = pygame.Surface((side, side), pygame.SRCALPHA)
    draw_mill_icon(surf, side // 2, side // 2, size_k, color_k, angle_k, MILL_STYLE_KIND)
    MILL_SURFACE_CACHE[key] = surf
    return surf


def blit_cached_mill(dest, cx, cy, size, color, angle):
    icon = get_cached_mill_surface(size, color, angle)
    rect = icon.get_rect(center=(int(cx), int(cy)))
    dest.blit(icon, rect)


# ═══════════════ DATA HELPERS ДЛЯ ВИЗУАЛИЗАТОРА ══════════════════════════════
def build_hero_day_summary(expanded, inst: GameData):
    """Собрать красивую сводку по герою и дням.

    Нужна для правой всплывающей таблицы по герою:
    - какие объекты он посещал в каждый день,
    - сколько очков хода суммарно потратил в этот день.

    Это уже не "официальная логика задачи", а просто удобная аналитика для UI.
    """
    out = {}
    if expanded.height == 0:
        return out

    def ensure_day(hid, day):
        out.setdefault(hid, {}).setdefault(day, {"route": [], "spent": 0})

    for row in expanded.iter_rows(named=True):
        hid = int(row["hero_id"])
        max_mp = int(inst.hero_mp_map.get(hid, 0))

        from_oid = int(row["object_id_from"])
        to_oid = int(row["object_id_to"])
        day_start = int(row["day_start"])
        day_arrive = int(row["day_arrive"])
        visit_day = int(row["day_leave"] if row["is_earlier"] else row["day_arrive"])

        dist = int(inst.get_distance(from_oid, to_oid))
        move_points_start = int(row["move_points_start"])
        move_points_arrive = int(row["move_points_arrive"])
        is_earlier = bool(row["is_earlier"])

        # Если объект достигнут в тот же день, весь путь пишем в этот день.
        # Если путь тянулся через ночь — раскладываем траты на два дня.
        if day_arrive == day_start:
            ensure_day(hid, day_start)
            out[hid][day_start]["spent"] += dist
        else:
            first_part = max(0, min(dist, move_points_start))
            second_part = max(0, dist - first_part)
            ensure_day(hid, day_start)
            ensure_day(hid, day_arrive)
            out[hid][day_start]["spent"] += first_part
            out[hid][day_arrive]["spent"] += second_part

        # Если герой приехал раньше, он ещё тратит очки на ожидание.
        if is_earlier:
            ensure_day(hid, day_arrive)
            out[hid][day_arrive]["spent"] += max(0, move_points_arrive)

            full_wait_days = max(0, visit_day - day_arrive - 1)
            for d in range(day_arrive + 1, visit_day):
                ensure_day(hid, d)
                out[hid][d]["spent"] += max_mp

            ensure_day(hid, visit_day)
            out[hid][visit_day]["spent"] += VISIT_COST
        else:
            visit_spend = VISIT_COST if move_points_arrive >= VISIT_COST else max(0, move_points_arrive)
            ensure_day(hid, visit_day)
            out[hid][visit_day]["spent"] += visit_spend

        ensure_day(hid, visit_day)
        out[hid][visit_day]["route"].append(to_oid)

    return out


def build_hero_journey(expanded):
    """Короткая сводка по герою:
    - шаги маршрута
    - общее золото
    """
    info = {}
    if expanded.height == 0:
        return info
    cur_h = None
    step = 0
    for row in expanded.iter_rows(named=True):
        hid = int(row["hero_id"])
        if hid != cur_h:
            cur_h = hid
            step = 0
        step += 1
        oid = int(row["object_id_to"])
        day = int(row["day_leave"])
        reward = int(row["reward"])
        if hid not in info:
            info[hid] = {"steps": [], "reward": 0}
        info[hid]["reward"] += reward
        info[hid]["steps"].append((step, oid, day, reward > 0))
    return info


def build_hero_segments(expanded):
    """Собрать отрезки маршрутов по героям."""
    segs = {}
    if expanded.height == 0:
        return segs
    for row in expanded.iter_rows(named=True):
        hid = int(row["hero_id"])
        segs.setdefault(hid, []).append((int(row["object_id_from"]), int(row["object_id_to"]), int(row["day_start"])))
    return segs


def build_hero_segment_details(expanded, inst: GameData):
    """Подробные данные по отрезкам маршрута.

    Нужны для:
    - подсветки линии на карте,
    - всплывающей подсказки по конкретному сегменту.
    """
    segs = {}
    if expanded.height == 0:
        return segs

    for row in expanded.iter_rows(named=True):
        hid = int(row["hero_id"])
        from_oid = int(row["object_id_from"])
        to_oid = int(row["object_id_to"])
        segs.setdefault(hid, []).append(
            {
                "hero_id": hid,
                "from_oid": from_oid,
                "to_oid": to_oid,
                "day_start": int(row["day_start"]),
                "day_arrive": int(row["day_arrive"]),
                "visit_day": int(row["day_leave"] if row["is_earlier"] else row["day_arrive"]),
                "distance": int(inst.get_distance(from_oid, to_oid)),
                "reward": int(row["reward"]),
                "is_earlier": bool(row["is_earlier"]),
                "is_late": bool(row["is_late"]),
                "move_points_start": int(row["move_points_start"]),
                "move_points_arrive": int(row["move_points_arrive"]),
                "move_points_leave": int(row["move_points_leave"]),
            }
        )
    return segs


def classify_hero_capacity(move_points: int):
    for upper, label, color in HERO_CAPACITY_CLASSES:
        if move_points < upper:
            return {"label": label, "color": color, "move_points": int(move_points)}
    _, label, color = HERO_CAPACITY_CLASSES[-1]
    return {"label": label, "color": color, "move_points": int(move_points)}


def build_hero_style_map(hero_mp_map):
    return {hid: classify_hero_capacity(mp) for hid, mp in hero_mp_map.items()}


def get_display_route_for_hero(hero_id, selected_day_filter, hero_segment_details):
    """Вытащить те сегменты маршрута, которые надо показать на карте.

    Если выбран день:
        показываем только часть маршрута, относящуюся к этому дню.
    Если день не выбран:
        показываем весь маршрут героя.
    """
    if hero_id is None:
        return set(), []

    full_details = hero_segment_details.get(hero_id, [])

    if selected_day_filter is None:
        oids = set()
        out_details = []
        for seg in full_details:
            if seg["from_oid"] != 0:
                oids.add(seg["from_oid"])
            oids.add(seg["to_oid"])
            seg2 = dict(seg)
            seg2["draw_day"] = seg["day_start"]
            out_details.append(seg2)
        return oids, out_details

    day = int(selected_day_filter)
    day_rows = [seg for seg in full_details if int(seg["visit_day"]) == day]
    if not day_rows:
        return set(), []

    oids = set()
    out_details = []

    first = True
    for seg in day_rows:
        oids.add(seg["to_oid"])
        if first:
            if not seg["is_earlier"]:
                if seg["from_oid"] != 0:
                    oids.add(seg["from_oid"])
                seg2 = dict(seg)
                seg2["draw_day"] = day
                out_details.append(seg2)
            first = False
        else:
            seg2 = dict(seg)
            seg2["draw_day"] = day
            out_details.append(seg2)

    return oids, out_details


# ═══════════════ TABLE TOOLTIPS ══════════════════════════════════════════════
def render_mill_table(font, oid, od_map, visits_map, inst: GameData, pinned):
    stripe = DAY_COLORS.get(od_map.get(oid, 0), (120, 120, 120))
    rows = [
        ("Мельница", str(oid), DARK_TEXT),
        ("День открытия", str(od_map[oid]), DARK_TEXT),
        ("От таверны", str(inst.dist_start_map.get(oid, "?")), DARK_TEXT),
    ]

    vis = visits_map.get(oid)
    if vis:
        visit_day_color = UNVISITED_RED if not vis.on_time else DARK_TEXT
        rows += [
            ("Герой", str(vis.hero_id), DARK_TEXT),
            ("День визита", str(vis.visit_day), visit_day_color),
        ]

    if pinned is not None and pinned != oid:
        d1 = int(inst.get_distance(pinned, oid))
        d2 = int(inst.get_distance(oid, pinned))
        if d1 == d2:
            rows.append((f"До #{pinned}", str(d1), DARK_TEXT))
        else:
            rows += [
                (f"→ #{pinned}", str(d2), DARK_TEXT),
                (f"← #{pinned}", str(d1), DARK_TEXT),
            ]

    CK, CV, ST = 145, 110, 5
    W = ST + CK + CV
    H = TBL_ROW_H * (1 + len(rows))
    surf = pygame.Surface((W, H), pygame.SRCALPHA)

    pygame.draw.rect(surf, TBL_HDR_BG, (ST, 0, W - ST, TBL_ROW_H))
    surf.blit(font.render("Параметр", True, TBL_HDR_TEXT), (ST + TBL_PAD, 3))
    surf.blit(font.render("Значение", True, TBL_HDR_TEXT), (ST + CK + TBL_PAD, 3))

    for i, (k, v, tc) in enumerate(rows):
        y = TBL_ROW_H * (i + 1)
        pygame.draw.rect(surf, TBL_ROW1 if i % 2 == 0 else TBL_ROW2, (ST, y, W - ST, TBL_ROW_H))
        surf.blit(font.render(k, True, DARK_TEXT), (ST + TBL_PAD, y + 3))
        surf.blit(font.render(v, True, tc), (ST + CK + TBL_PAD, y + 3))

    pygame.draw.rect(surf, stripe, (0, 0, ST, H))
    pygame.draw.rect(surf, TBL_BORDER, (0, 0, W, H), 1)
    pygame.draw.line(surf, TBL_BORDER, (ST + CK, 0), (ST + CK, H - 1))
    for i in range(len(rows) + 1):
        pygame.draw.line(surf, TBL_BORDER, (ST, TBL_ROW_H * i), (W - 1, TBL_ROW_H * i))

    return surf


def render_tavern_table(font, active_heroes_count: int, visited_count: int, score: int):
    stripe = TAV_WALL
    rows = [
        ("Локация", "Таверна", DARK_TEXT),
        ("Героев в решении", str(active_heroes_count), DARK_TEXT),
        ("Посещено объектов", str(visited_count), DARK_TEXT),
        ("Счёт", str(score), DARK_TEXT),
    ]

    CK, CV, ST = 150, 120, 5
    W = ST + CK + CV
    H = TBL_ROW_H * (1 + len(rows))
    surf = pygame.Surface((W, H), pygame.SRCALPHA)

    pygame.draw.rect(surf, TBL_HDR_BG, (ST, 0, W - ST, TBL_ROW_H))
    surf.blit(font.render("Параметр", True, TBL_HDR_TEXT), (ST + TBL_PAD, 3))
    surf.blit(font.render("Значение", True, TBL_HDR_TEXT), (ST + CK + TBL_PAD, 3))

    for i, (k, v, tc) in enumerate(rows):
        y = TBL_ROW_H * (i + 1)
        pygame.draw.rect(surf, TBL_ROW1 if i % 2 == 0 else TBL_ROW2, (ST, y, W - ST, TBL_ROW_H))
        surf.blit(font.render(k, True, DARK_TEXT), (ST + TBL_PAD, y + 3))
        surf.blit(font.render(v, True, tc), (ST + CK + TBL_PAD, y + 3))

    pygame.draw.rect(surf, stripe, (0, 0, ST, H))
    pygame.draw.rect(surf, TBL_BORDER, (0, 0, W, H), 1)
    pygame.draw.line(surf, TBL_BORDER, (ST + CK, 0), (ST + CK, H - 1))
    for i in range(len(rows) + 1):
        pygame.draw.line(surf, TBL_BORDER, (ST, TBL_ROW_H * i), (W - 1, TBL_ROW_H * i))

    return surf


def render_segment_table(font, seg):
    stripe = DAY_COLORS.get(seg["draw_day"], (120, 120, 120))
    status = "Вовремя"
    status_color = OK_GREEN
    if seg["is_earlier"]:
        status = "Рано"
        status_color = DARK_TEXT
    elif seg["is_late"]:
        status = "Поздно"
        status_color = UNVISITED_RED

    from_txt = "Таверна" if seg["from_oid"] == 0 else f"#{seg['from_oid']}"
    to_txt = f"#{seg['to_oid']}"

    rows = [
        ("Герой", str(seg["hero_id"]), DARK_TEXT),
        ("Откуда", from_txt, DARK_TEXT),
        ("Куда", to_txt, DARK_TEXT),
        ("День старта", str(seg["day_start"]), DARK_TEXT),
        ("День прибытия", str(seg["day_arrive"]), DARK_TEXT),
        ("День визита", str(seg["visit_day"]), DARK_TEXT),
        ("Дист.", str(seg["distance"]), DARK_TEXT),
        ("Статус", status, status_color),
    ]

    CK, CV, ST = 145, 118, 5
    W = ST + CK + CV
    H = TBL_ROW_H * (1 + len(rows))
    surf = pygame.Surface((W, H), pygame.SRCALPHA)

    pygame.draw.rect(surf, TBL_HDR_BG, (ST, 0, W - ST, TBL_ROW_H))
    surf.blit(font.render("Параметр", True, TBL_HDR_TEXT), (ST + TBL_PAD, 3))
    surf.blit(font.render("Значение", True, TBL_HDR_TEXT), (ST + CK + TBL_PAD, 3))

    for i, (k, v, tc) in enumerate(rows):
        y = TBL_ROW_H * (i + 1)
        pygame.draw.rect(surf, TBL_ROW1 if i % 2 == 0 else TBL_ROW2, (ST, y, W - ST, TBL_ROW_H))
        surf.blit(font.render(k, True, DARK_TEXT), (ST + TBL_PAD, y + 3))
        surf.blit(font.render(v, True, tc), (ST + CK + TBL_PAD, y + 3))

    pygame.draw.rect(surf, stripe, (0, 0, ST, H))
    pygame.draw.rect(surf, TBL_BORDER, (0, 0, W, H), 1)
    pygame.draw.line(surf, TBL_BORDER, (ST + CK, 0), (ST + CK, H - 1))
    for i in range(len(rows) + 1):
        pygame.draw.line(surf, TBL_BORDER, (ST, TBL_ROW_H * i), (W - 1, TBL_ROW_H * i))

    return surf


def ellipsize_text(text: str, font, max_width: int) -> str:
    """Обрезать длинный текст, чтобы он влез в указанную ширину."""
    if font.size(text)[0] <= max_width:
        return text

    ell = "…"
    lo, hi = 0, len(text)
    best = ell
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        if font.size(cand)[0] <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def render_hero_table_compact(font_title, font_cell, hid, hero_style_map, hero_day_summary, journey_info):
    style = hero_style_map.get(hid, {"label": "?", "color": STONE_LT, "move_points": 0})
    hc = style["color"]
    mp = style["move_points"]
    ji = journey_info.get(hid)
    total_r = ji["reward"] if ji else 0
    title = f"Герой #{hid}  ОД={mp}  золото={total_r}"

    day_map = hero_day_summary.get(hid, {})
    rows = []
    for day in range(1, 8):
        info = day_map.get(day, {"route": [], "spent": 0})
        route_str = " -> ".join(str(x) for x in info["route"]) if info["route"] else "—"
        spent_str = str(info["spent"])
        rows.append((str(day), route_str, spent_str))

    ST = 5
    C1 = 54
    C3 = 92
    route_width_raw = max(font_cell.size(r[1])[0] for r in rows) if rows else 180
    C2 = min(max(230, route_width_raw + 16), 620)
    W = ST + C1 + C2 + C3
    TITLE_H = 30
    HDR_H = 26
    RH = 24
    H = TITLE_H + HDR_H + RH * len(rows)

    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    surf.fill(PANEL_BG + (245,))

    pygame.draw.rect(surf, hc, (0, 0, W, TITLE_H))
    surf.blit(font_title.render(title, True, (255, 255, 255)), (ST + TBL_PAD, 5))

    y = TITLE_H
    pygame.draw.rect(surf, TBL_HDR_BG, (ST, y, W - ST, HDR_H))
    surf.blit(font_cell.render("День", True, TBL_HDR_TEXT), (ST + TBL_PAD, y + 3))
    surf.blit(font_cell.render("Маршрут", True, TBL_HDR_TEXT), (ST + C1 + TBL_PAD, y + 3))
    surf.blit(font_cell.render("Потрачено", True, TBL_HDR_TEXT), (ST + C1 + C2 + TBL_PAD, y + 3))

    for i, (day_txt, route_txt, spent_txt) in enumerate(rows):
        y = TITLE_H + HDR_H + RH * i
        bg = TBL_ROW1 if i % 2 == 0 else TBL_ROW2
        dc = DAY_COLORS.get(i + 1, (128, 128, 128))
        bg = tuple(int(b * 0.84 + d * 0.16) for b, d in zip(bg, dc))
        pygame.draw.rect(surf, bg, (ST, y, W - ST, RH))
        pygame.draw.rect(surf, dc, (0, y, ST, RH))

        route_fit = ellipsize_text(route_txt, font_cell, C2 - 2 * TBL_PAD)
        surf.blit(font_cell.render(day_txt, True, DARK_TEXT), (ST + TBL_PAD, y + 2))
        surf.blit(font_cell.render(route_fit, True, DARK_TEXT), (ST + C1 + TBL_PAD, y + 2))
        surf.blit(font_cell.render(spent_txt, True, DARK_TEXT), (ST + C1 + C2 + TBL_PAD, y + 2))

    pygame.draw.rect(surf, hc, (0, 0, ST, TITLE_H))
    pygame.draw.rect(surf, TBL_BORDER, (0, 0, W, H), 1)
    pygame.draw.line(surf, TBL_BORDER, (0, TITLE_H), (W - 1, TITLE_H))
    pygame.draw.line(surf, TBL_BORDER, (ST, TITLE_H + HDR_H), (W - 1, TITLE_H + HDR_H))
    pygame.draw.line(surf, TBL_BORDER, (ST + C1, TITLE_H), (ST + C1, H - 1))
    pygame.draw.line(surf, TBL_BORDER, (ST + C1 + C2, TITLE_H), (ST + C1 + C2, H - 1))

    for i in range(len(rows) + 1):
        yy = TITLE_H + HDR_H + RH * i
        if yy < H:
            pygame.draw.line(surf, TBL_BORDER, (ST, yy), (W - 1, yy))

    return surf


# ═══════════════ FRAME ═══════════════════════════════════════════════════════
def draw_homm3_frame(surf, w, h):
    """Нарисовать декоративную рамку в стиле HOMM-панелей."""
    t = FRAME_T
    pygame.draw.rect(surf, STONE_OUTER, (0, 0, w, h), 3)
    for r in [
        (3, 3, w - 6, t - 3),
        (3, h - t, w - 6, t - 3),
        (3, t, t - 3, h - 2 * t),
        (w - t, t, t - 3, h - 2 * t),
    ]:
        pygame.draw.rect(surf, STONE, r)

    pygame.draw.line(surf, STONE_HI, (3, 3), (w - 4, 3))
    pygame.draw.line(surf, STONE_LT, (4, 4), (w - 5, 4))
    pygame.draw.line(surf, STONE_HI, (3, 4), (3, h - 4))
    pygame.draw.line(surf, STONE_LT, (4, 5), (4, h - 5))
    pygame.draw.line(surf, STONE_DK, (4, h - 4), (w - 4, h - 4))
    pygame.draw.line(surf, STONE_DK, (w - 4, 4), (w - 4, h - 5))

    bk = 44
    for x in range(bk, w, bk):
        pygame.draw.line(surf, STONE_DK, (x, 5), (x, t - 2))
        pygame.draw.line(surf, STONE_DK, (x, h - t + 1), (x, h - 5))
    for y in range(t + bk, h - t, bk):
        pygame.draw.line(surf, STONE_DK, (5, y), (t - 2, y))
        pygame.draw.line(surf, STONE_DK, (w - t + 1, y), (w - 5, y))

    hm = t // 2
    for a, b in [
        ((5, hm), (w - 6, hm)),
        ((5, h - hm), (w - 6, h - hm)),
        ((hm, t), (hm, h - t)),
        ((w - hm, t), (w - hm, h - t)),
    ]:
        pygame.draw.line(surf, STONE_DK, a, b)

    pygame.draw.rect(surf, GOLD_DK, (t - 1, t - 1, w - 2 * t + 2, h - 2 * t + 2), 1)
    pygame.draw.rect(surf, GOLD, (t, t, w - 2 * t, h - 2 * t), 1)

    cs = t + 10
    for ir, ib in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        cx_ = (w - cs) if ir else 0
        cy_ = (h - cs) if ib else 0
        pygame.draw.rect(surf, STONE, (cx_ + 1, cy_ + 1, cs - 2, cs - 2))
        pygame.draw.line(surf, STONE_HI, (cx_ + 1, cy_ + 1), (cx_ + cs - 2, cy_ + 1))
        pygame.draw.line(surf, STONE_HI, (cx_ + 1, cy_ + 2), (cx_ + 1, cy_ + cs - 2))
        pygame.draw.line(surf, STONE_DK, (cx_ + cs - 2, cy_ + 1), (cx_ + cs - 2, cy_ + cs - 2))
        pygame.draw.line(surf, STONE_DK, (cx_ + 1, cy_ + cs - 2), (cx_ + cs - 2, cy_ + cs - 2))
        pygame.draw.rect(surf, GOLD, (cx_ + 5, cy_ + 5, cs - 10, cs - 10), 1)
        gx, gy = cx_ + cs // 2, cy_ + cs // 2
        gr = max(4, cs // 6)
        pygame.draw.circle(surf, GEM_RED, (gx, gy), gr)
        pygame.draw.circle(surf, GEM_HI, (gx - 1, gy - 1), max(1, gr // 2))


# ═══════════════ VIEW HELPERS ════════════════════════════════════════════════
def compute_fit_view(wxy, sw, sh, pad=FIT_PADDING, zoom_max=ZOOM_MAX_HARD):
    """Посчитать такой zoom и offset, чтобы вся карта влезла в окно."""
    wxy = ensure_finite_xy(wxy)
    if not len(wxy):
        return 1.0, np.zeros(2)

    lo, hi = wxy.min(0), wxy.max(0)
    sp = np.maximum(hi - lo, 1e-9)

    z = min((sw - 2 * pad) / sp[0], (sh - 2 * pad) / sp[1])
    if not np.isfinite(z) or z <= 0:
        z = 1.0

    z = max(ZOOM_MIN_HARD, min(z, zoom_max))
    offset = np.array([sw / 2.0, sh / 2.0]) - (lo + hi) / 2.0 * z
    if not np.isfinite(offset).all():
        offset = np.zeros(2)

    return z, offset


def compute_center_offset(wxy, sw, sh, zoom):
    wxy = ensure_finite_xy(wxy)
    if not len(wxy):
        return np.zeros(2)
    lo, hi = wxy.min(0), wxy.max(0)
    center = (lo + hi) / 2.0
    offset = np.array([sw / 2.0, sh / 2.0], dtype=float) - center * float(zoom)
    if not np.isfinite(offset).all():
        return np.zeros(2)
    return offset


def w2s(wxy, z, o):
    """World -> Screen: мировые координаты в экранные."""
    wxy = ensure_finite_xy(wxy)
    z = float(z) if np.isfinite(z) else 1.0
    o = np.asarray(o, dtype=float)

    if o.shape != (2,) or not np.isfinite(o).all():
        o = np.zeros(2)

    return wxy * z + o


def s2w(sxy, z, o):
    """Screen -> World: экранные координаты обратно в мировые."""
    z = float(z) if np.isfinite(z) and z != 0 else 1.0
    o = np.asarray(o, dtype=float)

    if o.shape != (2,) or not np.isfinite(o).all():
        o = np.zeros(2)

    return (np.array(sxy, dtype=float) - o) / z


def draw_text(surf, txt, pos, font, color=DARK_TEXT):
    surf.blit(font.render(txt, True, color), pos)


def draw_mill_node(
    surf,
    oid,
    x,
    y,
    iscl,
    bright,
    hovered,
    spin_angle,
    has_solution,
    all_route_objects,
    visits,
    od_map,
):
    """Нарисовать одну мельницу.

    bright = яркая ли она сейчас:
    - если объект входит в текущую подсветку, рисуем ярко,
    - если нет — слегка бледно.
    """
    color = DAY_COLORS.get(od_map[oid], (120, 120, 120)) if bright else INACTIVE_MILL_GRAY

    vis = visits.get(oid)
    base_sz = 20 if vis else 14
    if not bright:
        base_sz = max(10, base_sz - 1)

    msz = max(6, int(base_sz * MILL_SIZE_SCALE * iscl))
    ang = spin_angle if hovered else 0.0
    blit_cached_mill(surf, x, y, msz, color, ang)

    # Если решение загружено, но этот объект никем не посещён,
    # можно обвести его красным кольцом.
    if has_solution and oid not in all_route_objects and bright:
        rr = msz + max(3, int(4 * iscl))
        pygame.draw.circle(surf, UNVISITED_RED, (int(x), int(y)), rr, max(1, int(2 * iscl)))


def draw_moving_grid(screen, w, h, zoom, offset):
    """Нарисовать сетку, которая двигается вместе с картой."""
    sz = GRID_STEP * zoom
    if sz < 8:
        return

    ox = offset[0] % sz
    oy = offset[1] % sz

    x = ox
    while x < w:
        pygame.draw.line(screen, GRID_COLOR, (int(x), 0), (int(x), h))
        x += sz

    y = oy
    while y < h:
        pygame.draw.line(screen, GRID_COLOR, (0, int(y)), (w, int(y)))
        y += sz


def icon_scale_factor(zoom):
    return max(0.6, min(math.sqrt(max(float(zoom), 0.01)), 2.5))


def objects_near_cursor(mpos, obj_sxy, obj_ids, max_d=18.0):
    """Найти объекты рядом с курсором."""
    if not len(obj_sxy):
        return []
    pts = ensure_finite_xy(obj_sxy)
    dists = np.sqrt(np.square(pts - np.array(mpos, dtype=float)).sum(1))
    mask = dists <= max_d
    indices = np.where(mask)[0]
    return [obj_ids[indices[i]] for i in np.argsort(dists[mask])]


def draw_dashed_line(surf, color, s, e, width=2, dash=10, gap=6):
    if not all(np.isfinite(v) for v in [s[0], s[1], e[0], e[1]]):
        return
    dx, dy = e[0] - s[0], e[1] - s[1]
    d = math.hypot(dx, dy)
    if d < 1:
        return
    ux, uy = dx / d, dy / d
    step = dash + gap
    for i in range(int(d / step) + 1):
        a = i * step
        b = min(a + dash, d)
        pygame.draw.line(
            surf,
            color,
            (int(s[0] + ux * a), int(s[1] + uy * a)),
            (int(s[0] + ux * b), int(s[1] + uy * b)),
            width,
        )


def measure_label_position(p1: np.ndarray, p2: np.ndarray, offset_px: float = 30.0) -> np.ndarray:
    """Поставить подпись расстояния чуть в стороне от линии."""
    mid = (p1 + p2) / 2.0
    d = p2 - p1
    n = np.array([-d[1], d[0]], dtype=float)
    nn = float(np.linalg.norm(n))
    if not np.isfinite(nn) or nn < 1e-6:
        return mid + np.array([0.0, -offset_px], dtype=float)

    n /= nn
    if n[1] > 0:
        n = -n
    pos = mid + n * offset_px
    pos[1] -= 6.0
    return pos


def point_segment_distance(p, a, b):
    """Расстояние от точки p до отрезка ab."""
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def hovered_route_segment(mpos, seg_list, sxy, node_to_idx, max_dist=8.0):
    """Найти отрезок маршрута, над которым сейчас курсор."""
    if not seg_list:
        return None

    best_seg = None
    best_dist = max_dist

    for seg in seg_list:
        if seg["from_oid"] not in node_to_idx or seg["to_oid"] not in node_to_idx:
            continue
        p1 = sxy[node_to_idx[seg["from_oid"]]]
        p2 = sxy[node_to_idx[seg["to_oid"]]]
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        d = point_segment_distance(np.array(mpos, dtype=float), p1, p2)
        if d <= best_dist:
            best_dist = d
            best_seg = seg

    return best_seg


def draw_day_legend(screen, font_cell, spin_angle, hovered_row, selected_row, day_total_counts, day_success_counts):
    """Левая легенда по дням недели."""
    lx, ly = FRAME_T + 12, FRAME_T + 52
    col_day = 88
    col_succ = 90
    col_total = 74
    hdr_h = 24
    row_h = 26

    rows = [(d, str(d), day_success_counts.get(d, 0), day_total_counts.get(d, 0)) for d in range(1, 8)]
    rows.append(
        (
            0,
            "Все",
            sum(day_success_counts.get(d, 0) for d in range(1, 8)),
            sum(day_total_counts.get(d, 0) for d in range(1, 8)),
        )
    )

    table_w = col_day + col_succ + col_total
    table_h = hdr_h + row_h * len(rows)

    pygame.draw.rect(screen, TBL_HDR_BG, (lx, ly, table_w, hdr_h))
    draw_text(screen, "День", (lx + TBL_PAD, ly + 3), font_cell, TBL_HDR_TEXT)
    draw_text(screen, "Успешно", (lx + col_day + TBL_PAD, ly + 3), font_cell, TBL_HDR_TEXT)
    draw_text(screen, "Всего", (lx + col_day + col_succ + TBL_PAD, ly + 3), font_cell, TBL_HDR_TEXT)

    rects = {}
    hover_tint = (215, 205, 180)
    all_row_light = (250, 246, 238)

    for idx, (key, label, succ, total) in enumerate(rows):
        y = ly + hdr_h + row_h * idx
        row_rect = pygame.Rect(lx, y, table_w, row_h)
        rects[key] = row_rect

        base_bg = TBL_ROW1 if idx % 2 == 0 else TBL_ROW2
        if key == 0:
            base_bg = blend_colors(base_bg, all_row_light, 0.35)

        stripe_color = TAV_WALL_LT if key == 0 else DAY_COLORS[key]

        if key == selected_row:
            bg = blend_colors(base_bg, stripe_color, 0.42)
        elif key == hovered_row:
            bg = blend_colors(base_bg, hover_tint, 0.45)
        else:
            bg = base_bg

        pygame.draw.rect(screen, bg, row_rect)

        if key == 0:
            pygame.draw.rect(screen, TAV_WALL, (lx, y, 6, row_h))
            draw_text(screen, label, (lx + 14, y + 3), font_cell, DARK_TEXT)
        else:
            pygame.draw.rect(screen, DAY_COLORS[key], (lx, y, 6, row_h))
            blit_cached_mill(screen, lx + 18, y + row_h // 2 + 1, max(6, int(11 * MILL_SIZE_SCALE)), DAY_COLORS[key], spin_angle)
            draw_text(screen, label, (lx + 34, y + 3), font_cell, DARK_TEXT)

        incomplete = succ < total
        succ_color = UNVISITED_RED if incomplete else (OK_GREEN if succ > 0 else DARK_TEXT)

        draw_text(screen, str(succ), (lx + col_day + TBL_PAD, y + 3), font_cell, succ_color)
        draw_text(screen, str(total), (lx + col_day + col_succ + TBL_PAD, y + 3), font_cell, DARK_TEXT)

    pygame.draw.rect(screen, TBL_BORDER, (lx, ly, table_w, table_h), 1)
    pygame.draw.line(screen, TBL_BORDER, (lx + col_day, ly), (lx + col_day, ly + table_h))
    pygame.draw.line(screen, TBL_BORDER, (lx + col_day + col_succ, ly), (lx + col_day + col_succ, ly + table_h))

    for i in range(len(rows) + 1):
        yy = ly + hdr_h + row_h * i
        pygame.draw.line(screen, TBL_BORDER, (lx, yy), (lx + table_w, yy))

    return rects


def draw_all_routes_cell(screen, rect, selected, hovered, font):
    """Ячейка 'Все' в панели героев."""
    generic_hover_tint = (215, 205, 180)
    base_bg = TBL_ROW1
    if selected:
        cell_bg = blend_colors(base_bg, TAV_WALL_LT, 0.42)
    elif hovered:
        cell_bg = blend_colors(base_bg, generic_hover_tint, 0.45)
    else:
        cell_bg = base_bg

    pygame.draw.rect(screen, cell_bg, rect, border_radius=4)
    pygame.draw.rect(screen, PANEL_BORDER, rect, 1, border_radius=4)

    cy = rect.centery
    xs = [rect.x + 11, rect.x + 19, rect.x + 27]
    cols = [(66, 125, 245), (50, 180, 70), (240, 145, 30)]
    for x, c in zip(xs, cols):
        pygame.draw.circle(screen, c, (x, cy), 4)
        pygame.draw.circle(screen, STONE_DK, (x, cy), 4, 1)

    label = font.render("Все", True, DARK_TEXT)
    lr = label.get_rect(midleft=(rect.x + 35, cy))
    screen.blit(label, lr)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--solution", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    validate_data_dir(data_dir)

    target_fps = max(1, int(args.fps))

    # 1) Загружаем справочные данные игры
    inst = GameData(data_dir)

    # 2) Создаем симулятор, который умеет по этим данным
    #    считать маршруты и score
    simulator = RouteSimulator(inst)

    # 3) Строим координаты для рисования карты
    world_xy, object_ids, node_to_idx = build_layout(inst)

    # 4) Загружаем решение
    raw = Solution.load(args.solution.resolve() if args.solution else None)

    # 5) Чистим решение по базовым правилам
    checked = raw.basic_check(inst) if raw.height else raw

    # 6) Разворачиваем в подробную таблицу переходов
    expanded = simulator.expand_solution(checked) if checked.height else pl.DataFrame()

    # 7) Считаем score
    score = simulator.evaluate_solution(raw) if raw.height else 0

    # 8) Строим вспомогательные структуры для интерфейса
    routes = checked.routes_by_hero() if checked.height else {}
    od_map = dict(inst.object_day_map)
    visits = visits_from_expanded(expanded, od_map)
    has_solution = raw.height > 0

    active_heroes = sorted(routes.keys())
    all_route_objects = set()
    for rt in routes.values():
        all_route_objects.update(rt)

    journey_info = build_hero_journey(expanded)
    hero_segments = build_hero_segments(expanded)
    hero_segment_details = build_hero_segment_details(expanded, inst)
    hero_day_summary = build_hero_day_summary(expanded, inst)
    hero_style_map = build_hero_style_map(inst.hero_mp_map)

    # Объекты по дням: пригодится для фильтра "покажи день N"
    day_objects: Dict[int, Set[int]] = {}
    for oid, d in od_map.items():
        day_objects.setdefault(d, set()).add(oid)

    day_total_counts = {d: len(day_objects.get(d, set())) for d in range(1, 8)}
    day_success_counts = {d: 0 for d in range(1, 8)}
    for vis in visits.values():
        if vis.on_time:
            day_success_counts[vis.open_day] += 1

    pygame.init()
    di = pygame.display.Info()
    cur_w = min(INIT_WIDTH, int(di.current_w * 0.80))
    cur_h = min(INIT_HEIGHT, int(di.current_h * 0.80))
    is_fs = False
    screen = pygame.display.set_mode((cur_w, cur_h), pygame.RESIZABLE)

    sol_name = args.solution.name if args.solution else "без решения"
    pygame.display.set_caption(f"Визуализатор Heroes VRPTW — {sol_name}")

    clock = pygame.time.Clock()

    f_sm = pygame.font.SysFont("arial", FONT_SM)
    f_md = pygame.font.SysFont("arial", FONT_MD)
    f_tbl_hero = pygame.font.SysFont("arial", max(19, int(round(FONT_TBL * 1.2))))
    f_sm_hero = pygame.font.SysFont("arial", max(22, int(round(FONT_SM * 1.2))))

    def compute_initial_view(sw, sh):
        """Посчитать стартовый вид карты."""
        fit_zoom, _ = compute_fit_view(world_xy, sw, sh, pad=FIT_PADDING, zoom_max=ZOOM_MAX_HARD)
        initial_zoom = max(ZOOM_MIN_HARD, min(ZOOM_MAX_HARD, fit_zoom * 0.90))
        initial_offset = compute_center_offset(world_xy, sw, sh, initial_zoom)
        zoom_min = max(ZOOM_MIN_HARD, initial_zoom / 2.0)
        zoom_max = ZOOM_MAX_HARD
        return initial_zoom, initial_offset, zoom_min, zoom_max

    zoom, offset, dynamic_zoom_min, dynamic_zoom_max = compute_initial_view(cur_w, cur_h)

    dragging = False
    last_mouse = (0, 0)

    # pinned_object — объект, который пользователь "заколол" правой кнопкой
    # для измерения расстояний до других объектов.
    pinned_object: Optional[int] = None

    # selected_day_filter:
    # None = показываем все дни
    # 1..7 = только выбранный день
    selected_day_filter: Optional[int] = None

    # selected_hero_filter:
    # None = не фиксируем конкретного героя
    selected_hero_filter: Optional[int] = None

    # show_all_hero_routes:
    # если True, одновременно подсвечиваем маршруты всех героев
    show_all_hero_routes = False

    btn_fs = UIButton("Во весь экран", 170, 33)
    btn_fit = UIButton("Показать всё", 135, 33)

    win_w, win_h = cur_w, cur_h
    t0 = _time.monotonic()
    day_legend_rects: Dict[int, pygame.Rect] = {}

    def clamp_zoom():
        nonlocal zoom
        if not np.isfinite(zoom) or zoom <= 0:
            zoom = 1.0
        zoom = max(dynamic_zoom_min, min(zoom, dynamic_zoom_max))

    def toggle_fs():
        """Переключить полноэкранный режим."""
        nonlocal is_fs, cur_w, cur_h, screen, zoom, offset, dynamic_zoom_min, dynamic_zoom_max, win_w, win_h
        is_fs = not is_fs
        if is_fs:
            win_w, win_h = cur_w, cur_h
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            cur_w, cur_h = screen.get_size()
        else:
            cur_w, cur_h = win_w, win_h
            screen = pygame.display.set_mode((cur_w, cur_h), pygame.RESIZABLE)

        zoom, offset, dynamic_zoom_min, dynamic_zoom_max = compute_initial_view(cur_w, cur_h)
        clamp_zoom()

    def do_fit():
        """Показать всю карту в окне."""
        nonlocal zoom, offset, dynamic_zoom_min, dynamic_zoom_max
        zoom, offset, dynamic_zoom_min, dynamic_zoom_max = compute_initial_view(cur_w, cur_h)
        clamp_zoom()

    hero_panel_rect = pygame.Rect(0, 0, 0, 0)
    hp_x = hp_y = hp_w = hp_h = 0

    def panel_cell_rect(index: int) -> pygame.Rect:
        """Прямоугольник ячейки героя в правой панели."""
        col = index // HERO_ROWS
        row = index % HERO_ROWS
        x = hp_x + 10 + col * (HERO_CELL_W + HERO_CELL_GAP)
        y = hp_y + 36 + row * (HERO_CELL_H + HERO_CELL_GAP)
        return pygame.Rect(x, y, HERO_CELL_W, HERO_CELL_H)

    def panel_item_at_pos(pos):
        """Определить, над чем сейчас курсор в панели героев."""
        if not active_heroes:
            return None
        total = len(active_heroes) + 1
        for idx in range(total):
            if panel_cell_rect(idx).collidepoint(pos):
                if idx < len(active_heroes):
                    return ("hero", active_heroes[idx], idx)
                return ("all", None, idx)
        return None

    running = True
    while running:
        clock.tick(target_fps)
        now = _time.monotonic() - t0
        mpos = pygame.mouse.get_pos()
        spin_angle = (now * BLADE_SPIN_SPEED) % 360.0

        # Размеры панели героев рассчитываем динамически,
        # потому что количество героев может быть разным.
        if active_heroes:
            cell_count = len(active_heroes) + 1
            nc = max(1, math.ceil(cell_count / HERO_ROWS))
            nr = min(cell_count, HERO_ROWS)
            grid_w = nc * HERO_CELL_W + max(0, nc - 1) * HERO_CELL_GAP
            grid_h = nr * HERO_CELL_H + max(0, nr - 1) * HERO_CELL_GAP
            hp_w = grid_w + 20
            hp_h = grid_h + 44
            hp_x = cur_w - FRAME_T - hp_w - 8
            hp_y = FRAME_T + 44
            hero_panel_rect = pygame.Rect(hp_x, hp_y, hp_w, hp_h)
        else:
            hero_panel_rect = pygame.Rect(0, 0, 0, 0)

        btn_fs.set_position(cur_w - FRAME_T - 180, FRAME_T + 4)
        btn_fit.set_position(cur_w - FRAME_T - 325, FRAME_T + 4)
        btn_fs.text = "Оконный режим" if is_fs else "Во весь экран"
        btn_fs.handle_motion(mpos)
        btn_fit.handle_motion(mpos)

        over_top_buttons = btn_fs.rect.collidepoint(mpos) or btn_fit.rect.collidepoint(mpos)

        # ── обработка событий ───────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.VIDEORESIZE and not is_fs:
                cur_w, cur_h = ev.w, ev.h
                screen = pygame.display.set_mode((cur_w, cur_h), pygame.RESIZABLE)
                do_fit()

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    # ESC работает как "снять текущий уровень выделения".
                    if pinned_object is not None:
                        pinned_object = None
                    elif selected_hero_filter is not None:
                        selected_hero_filter = None
                    elif show_all_hero_routes:
                        show_all_hero_routes = False
                    elif selected_day_filter is not None:
                        selected_day_filter = None
                    elif is_fs:
                        toggle_fs()
                    else:
                        running = False
                elif ev.key == pygame.K_F11:
                    toggle_fs()
                elif ev.key in (pygame.K_f, pygame.K_r):
                    do_fit()

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    legend_clicked = next((d for d, r in day_legend_rects.items() if r.collidepoint(ev.pos)), None)
                    panel_clicked = panel_item_at_pos(ev.pos)

                    if btn_fs.is_clicked(ev.pos):
                        toggle_fs()
                    elif btn_fit.is_clicked(ev.pos):
                        do_fit()
                    elif legend_clicked is not None:
                        # Клик по легенде дней:
                        # "Все" -> снимаем фильтр, иначе выбираем день
                        selected_day_filter = None if legend_clicked == 0 else legend_clicked
                    elif panel_clicked is not None and panel_clicked[0] == "all":
                        # Клик по ячейке "Все" в панели героев
                        if show_all_hero_routes and selected_hero_filter is None:
                            show_all_hero_routes = False
                        else:
                            selected_hero_filter = None
                            show_all_hero_routes = True
                    elif panel_clicked is not None and panel_clicked[0] == "hero":
                        # Клик по конкретному герою
                        hero_clicked = panel_clicked[1]
                        selected_hero_filter = None if hero_clicked == selected_hero_filter else hero_clicked
                        show_all_hero_routes = False
                    elif hero_panel_rect.collidepoint(ev.pos):
                        pass
                    else:
                        # Иначе начинаем перетаскивать карту
                        dragging = True
                        last_mouse = ev.pos

                elif ev.button == 3:
                    # Правая кнопка мыши:
                    # если нажали на объект, закрепляем/снимаем его как "точку линейки"
                    nb = objects_near_cursor(ev.pos, w2s(world_xy, zoom, offset)[1:], object_ids, 20.0)
                    if nb:
                        cl = nb[0]
                        pinned_object = None if cl == pinned_object else cl
                    else:
                        pinned_object = None

                elif ev.button in (4, 5):
                    # Старый формат колеса мыши (некоторые системы)
                    m = np.array(ev.pos, dtype=float)
                    b = s2w(ev.pos, zoom, offset)
                    zoom *= 1.1 if ev.button == 4 else 1 / 1.1
                    clamp_zoom()
                    offset = m - b * zoom

            elif ev.type == pygame.MOUSEWHEEL:
                # Новый формат колеса мыши
                m = np.array(mpos, dtype=float)
                b = s2w(tuple(m), zoom, offset)
                if ev.y > 0:
                    zoom *= 1.1
                elif ev.y < 0:
                    zoom *= 1 / 1.1
                clamp_zoom()
                offset = m - b * zoom

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False

            elif ev.type == pygame.MOUSEMOTION and dragging:
                offset += np.array([ev.pos[0] - last_mouse[0], ev.pos[1] - last_mouse[1]], dtype=float)
                if not np.isfinite(offset).all():
                    offset = np.zeros(2)
                last_mouse = ev.pos

        # ── вычисляем, что сейчас под курсором и что подсвечивать ───────────
        on_hero_ui = hero_panel_rect.collidepoint(mpos)
        on_ui_block = on_hero_ui or over_top_buttons

        panel_hover = panel_item_at_pos(mpos) if hero_panel_rect.collidepoint(mpos) else None
        hovered_hero = panel_hover[1] if panel_hover is not None and panel_hover[0] == "hero" else None
        hovered_all_cell = bool(panel_hover is not None and panel_hover[0] == "all")

        iscl = icon_scale_factor(zoom)
        detect_r = max(18.0, 14.0 * iscl)
        sxy = w2s(world_xy, zoom, offset)
        obj_sxy = sxy[1:]
        hov_list = [] if on_ui_block else objects_near_cursor(mpos, obj_sxy, object_ids, detect_r)
        hov_set = set(hov_list)

        tav_hovered = False
        tav_pos = sxy[0]
        tav_sz = max(10, int(30 * iscl))
        if not on_ui_block and np.isfinite(tav_pos).all():
            tav_r = max(22.0, tav_sz * 1.2)
            tav_hovered = float(np.linalg.norm(np.array(mpos, dtype=float) - tav_pos)) <= tav_r

        hovered_day: Optional[int] = None
        for d, r in day_legend_rects.items():
            if r.collidepoint(mpos):
                hovered_day = d
                break

        display_route_oids: Set[int] = set()
        display_route_details = []

        if selected_hero_filter is not None:
            display_route_oids, display_route_details = get_display_route_for_hero(
                selected_hero_filter,
                selected_day_filter,
                hero_segment_details,
            )
        elif show_all_hero_routes or hovered_all_cell:
            for hid in active_heroes:
                oids_h, det_h = get_display_route_for_hero(hid, selected_day_filter, hero_segment_details)
                display_route_oids |= oids_h
                display_route_details.extend(det_h)
        elif hovered_hero is not None:
            display_route_oids, display_route_details = get_display_route_for_hero(
                hovered_hero,
                selected_day_filter,
                hero_segment_details,
            )

        highlight_oids: Optional[Set[int]] = None
        selected_sets = []
        if selected_day_filter is not None:
            selected_sets.append(set(day_objects.get(selected_day_filter, set())))
        if selected_hero_filter is not None:
            selected_sets.append(display_route_oids)
        elif show_all_hero_routes:
            selected_sets.append(display_route_oids)

        if selected_sets:
            highlight_oids = set().union(*selected_sets)
        elif hovered_day is not None and hovered_day > 0:
            highlight_oids = day_objects.get(hovered_day, set())
        elif hovered_hero is not None:
            highlight_oids = display_route_oids
        elif hovered_all_cell:
            highlight_oids = display_route_oids

        force_bright_oids = set(hov_set)
        if pinned_object is not None:
            force_bright_oids.add(pinned_object)
        force_bright_oids.update(display_route_oids)

        hovered_segment = None
        if display_route_details and not on_ui_block and not tav_hovered:
            hovered_segment = hovered_route_segment(
                mpos,
                display_route_details,
                sxy,
                node_to_idx,
                max_dist=max(7.0, 6.0 * iscl),
            )

        # ── рисуем основную сцену ────────────────────────────────────────────
        screen.fill(BG_COLOR)
        draw_moving_grid(screen, cur_w, cur_h, zoom, offset)

        # Сначала таверна
        if tav_hovered:
            draw_tavern_smoke(screen, tav_pos[0], tav_pos[1], tav_sz, now)
        draw_tavern_icon(screen, tav_pos[0], tav_pos[1], tav_sz)

        # Потом объекты и маршруты
        if display_route_details and display_route_oids:
            # Сначала рисуем все НЕмаршрутные объекты,
            # потом поверх — линии маршрутов и сами маршрутные объекты.
            for bright_pass in (False, True):
                for oid in object_ids:
                    if oid in display_route_oids:
                        continue

                    base_bright = (highlight_oids is None) or (oid in highlight_oids)
                    bright = base_bright or (oid in force_bright_oids)
                    if bright != bright_pass:
                        continue

                    ix = node_to_idx[oid]
                    x, y = sxy[ix]
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue

                    draw_mill_node(
                        screen,
                        oid,
                        x,
                        y,
                        iscl,
                        bright,
                        oid in hov_set,
                        spin_angle,
                        has_solution,
                        all_route_objects,
                        visits,
                        od_map,
                    )

            # Линии маршрута
            for seg in display_route_details:
                p1 = sxy[node_to_idx[seg["from_oid"]]]
                p2 = sxy[node_to_idx[seg["to_oid"]]]
                if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                    continue
                seg_c = DAY_COLORS.get(seg["draw_day"], (128, 128, 128))
                pygame.draw.line(screen, (30, 25, 18), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 5)
                pygame.draw.line(screen, seg_c, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 3)

            # Подсветка сегмента под курсором
            if hovered_segment is not None:
                p1 = sxy[node_to_idx[hovered_segment["from_oid"]]]
                p2 = sxy[node_to_idx[hovered_segment["to_oid"]]]
                pygame.draw.line(screen, GOLD, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 2)

            # Маршрутные объекты рисуем поверх
            for oid in object_ids:
                if oid not in display_route_oids:
                    continue

                ix = node_to_idx[oid]
                x, y = sxy[ix]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue

                draw_mill_node(
                    screen,
                    oid,
                    x,
                    y,
                    iscl,
                    True,
                    oid in hov_set,
                    spin_angle,
                    has_solution,
                    all_route_objects,
                    visits,
                    od_map,
                )
        else:
            # Обычный режим: без маршрутов на переднем плане
            for bright_pass in (False, True):
                for oid in object_ids:
                    base_bright = (highlight_oids is None) or (oid in highlight_oids)
                    bright = base_bright or (oid in force_bright_oids)
                    if bright != bright_pass:
                        continue

                    ix = node_to_idx[oid]
                    x, y = sxy[ix]
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue

                    draw_mill_node(
                        screen,
                        oid,
                        x,
                        y,
                        iscl,
                        bright,
                        oid in hov_set,
                        spin_angle,
                        has_solution,
                        all_route_objects,
                        visits,
                        od_map,
                    )

        # ── верхняя строка состояния ────────────────────────────────────────
        sp_w = cur_w - 2 * FRAME_T
        sp_h = 40
        sp = pygame.Surface((sp_w, sp_h), pygame.SRCALPHA)
        sp.fill(STATUS_BG + (220,))
        screen.blit(sp, (FRAME_T, FRAME_T))
        pygame.draw.rect(screen, GOLD_DK, (FRAME_T, FRAME_T, sp_w, sp_h), 1)

        day_mode_txt = "Все дни" if selected_day_filter is None else f"День {selected_day_filter}"
        if selected_hero_filter is not None:
            hero_mode_txt = f"Герой {selected_hero_filter}"
        elif show_all_hero_routes or hovered_all_cell:
            hero_mode_txt = "Все герои"
        else:
            hero_mode_txt = "Все герои"

        pin_s = f"  метка=#{pinned_object}" if pinned_object else ""
        status = (
            f"Счёт={score}  Посещено={len(visits)}  "
            f"{day_mode_txt}  {hero_mode_txt}{pin_s}  |  F:в кадр  F11:экран  ПКМ:линейка  ESC:сброс"
        )
        status_lbl = f_md.render(status, True, PARCHMENT)
        status_rect = status_lbl.get_rect()
        status_rect.x = FRAME_T + 10
        status_rect.centery = FRAME_T + sp_h // 2
        screen.blit(status_lbl, status_rect)

        # ── легенда дней ────────────────────────────────────────────────────
        selected_legend_row = 0 if selected_day_filter is None else selected_day_filter
        day_legend_rects = draw_day_legend(
            screen,
            f_sm,
            spin_angle,
            hovered_day,
            selected_legend_row,
            day_total_counts,
            day_success_counts,
        )

        # ── панель героев ───────────────────────────────────────────────────
        if active_heroes:
            panel = pygame.Surface((hp_w, hp_h), pygame.SRCALPHA)
            panel.fill(PANEL_BG + (215,))
            pygame.draw.rect(panel, PANEL_BORDER, panel.get_rect(), 1)
            screen.blit(panel, (hp_x, hp_y))
            draw_text(screen, "Герои", (hp_x + 10, hp_y + 8), f_md, DARK_TEXT)

            generic_hover_tint = (215, 205, 180)

            for idx, hid in enumerate(active_heroes):
                hstyle = hero_style_map.get(hid, {"color": STONE_LT})
                hcolor = hstyle["color"]
                cr = panel_cell_rect(idx)

                cell_base = TBL_ROW1 if idx % 2 == 0 else TBL_ROW2
                if hid == selected_hero_filter:
                    cell_bg = blend_colors(cell_base, hcolor, 0.42)
                elif hid == hovered_hero and selected_hero_filter is None and not show_all_hero_routes:
                    cell_bg = blend_colors(cell_base, generic_hover_tint, 0.45)
                else:
                    cell_bg = cell_base

                pygame.draw.rect(screen, cell_bg, cr, border_radius=4)
                pygame.draw.rect(screen, PANEL_BORDER, cr, 1, border_radius=4)

                cx_ = cr.x + HERO_ICON_SZ // 2 + 6
                cy_ = cr.y + HERO_CELL_H // 2
                draw_hero_figure(screen, cx_, cy_, HERO_ICON_SZ + 2, hcolor)
                draw_text(screen, str(hid), (cx_ + HERO_ICON_SZ // 2 + 4, cy_ - 9), f_sm, DARK_TEXT)

            all_idx = len(active_heroes)
            all_rect = panel_cell_rect(all_idx)
            selected_all = show_all_hero_routes and selected_hero_filter is None
            draw_all_routes_cell(screen, all_rect, selected_all, hovered_all_cell and not selected_all, f_sm)

        # ── кнопки сверху ───────────────────────────────────────────────────
        btn_fs.draw(screen, f_md)
        btn_fit.draw(screen, f_md)

        # ── всплывающие подсказки ───────────────────────────────────────────
        GAP = 8
        if hovered_hero is not None:
            tip = render_hero_table_compact(
                f_sm_hero,
                f_tbl_hero,
                hovered_hero,
                hero_style_map,
                hero_day_summary,
                journey_info,
            )
            tw, th = tip.get_size()
            tx = mpos[0] - tw - 18 if mpos[0] + 18 + tw > cur_w - FRAME_T else mpos[0] + 18
            ty = max(FRAME_T + 4, min(mpos[1] + 18, cur_h - FRAME_T - th - 4))
            screen.blit(tip, (tx, ty))

        elif tav_hovered:
            tip = render_tavern_table(f_sm, len(active_heroes), len(visits), score)
            tw, th = tip.get_size()
            tx = mpos[0] - tw - 18 if mpos[0] + 18 + tw > cur_w - FRAME_T else mpos[0] + 18
            ty = max(FRAME_T + 4, min(mpos[1] + 18, cur_h - FRAME_T - th - 4))
            screen.blit(tip, (tx, ty))

        elif hov_list or hovered_segment is not None:
            surfs = []

            if hov_list:
                surfs.extend([render_mill_table(f_sm, oid, od_map, visits, inst, pinned_object) for oid in hov_list])

            if hovered_segment is not None:
                surfs.append(render_segment_table(f_sm, hovered_segment))

            total_h = sum(s.get_height() for s in surfs) + GAP * (len(surfs) - 1)
            tw = max(s.get_width() for s in surfs)
            tx = mpos[0] + 18
            ty = mpos[1] + 18
            if tx + tw > cur_w - FRAME_T - 4:
                tx = mpos[0] - tw - 18
            if ty + total_h > cur_h - FRAME_T - 4:
                ty = max(FRAME_T + 4, cur_h - FRAME_T - 4 - total_h)

            cy_ = ty
            for s in surfs:
                screen.blit(s, (tx, cy_))
                cy_ += s.get_height() + GAP

        # ── линейка расстояний от "заколотой" точки ────────────────────────
        if pinned_object is not None:
            for ho in hov_list:
                if ho == pinned_object:
                    continue
                p1 = sxy[node_to_idx[pinned_object]]
                p2 = sxy[node_to_idx[ho]]
                draw_dashed_line(screen, MEASURE_LINE_COLOR, (p1[0], p1[1]), (p2[0], p2[1]))
                md = int(inst.get_distance(pinned_object, ho))
                pos = measure_label_position(p1, p2, offset_px=30.0)
                if np.isfinite(pos).all():
                    lbl = f_md.render(str(md), True, MEASURE_COLOR)
                    lr = lbl.get_rect(center=(int(pos[0]), int(pos[1])))
                    bgr = lr.inflate(14, 8)
                    pygame.draw.rect(screen, MEASURE_TEXT_BG, bgr, border_radius=4)
                    pygame.draw.rect(screen, MEASURE_COLOR, bgr, 1, border_radius=4)
                    screen.blit(lbl, lr)

        # Подсветка "заколотой" точки
        if pinned_object is not None:
            pidx = node_to_idx[pinned_object]
            px, py = sxy[pidx]
            if np.isfinite(px) and np.isfinite(py):
                px, py = int(px), int(py)
                pr = max(14, int(22 * iscl))
                pygame.draw.circle(screen, PIN_RING_COLOR, (px, py), pr, 2)
                for dd in [(-pr, 0), (pr, 0), (0, -pr), (0, pr)]:
                    pygame.draw.line(
                        screen,
                        PIN_RING_COLOR,
                        (px + dd[0] // 2, py + dd[1] // 2),
                        (px + dd[0], py + dd[1]),
                        1,
                    )
                screen.blit(f_sm.render(f"#{pinned_object}", True, PIN_RING_COLOR), (px + pr + 5, py - pr))

        # Рамку рисуем в самом конце, чтобы она была поверх всего
        draw_homm3_frame(screen, cur_w, cur_h)

        pygame.display.flip()

    pygame.quit()


# ═══════════════ ПОМОЩНИКИ ДЛЯ ПОДГОТОВКИ ДАННЫХ В UI ═══════════════════════
def visits_from_expanded(df, od):
    """Построить словарь visit info по объектам.

    object_id -> VisitInfo

    Это удобная структура для:
    - подсказок по мельнице,
    - подсчета успешных посещений по дням,
    - показа статуса "вовремя/поздно".
    """
    v = {}
    if df.height == 0:
        return v
    for row in df.iter_rows(named=True):
        oid = int(row["object_id_to"])
        early = bool(row["is_earlier"])
        late = bool(row["is_late"])
        vd = int(row["day_leave"] if early else row["day_arrive"])
        o = int(od[oid])
        v[oid] = VisitInfo(int(row["hero_id"]), vd, o, int(row["reward"]), vd == o, early, late)
    return v


if __name__ == "__main__":
    main()
