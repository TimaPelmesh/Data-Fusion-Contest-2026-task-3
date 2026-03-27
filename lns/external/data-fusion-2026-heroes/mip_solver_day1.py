#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Попытка импортировать библиотеку PuLP.
# Именно через неё мы будем строить MIP-модель и передавать её
# внешнему солверу (CBC / HiGHS / Gurobi).
# ------------------------------------------------------------
try:
    import pulp
    from pulp import (
        LpBinary,
        LpContinuous,
        LpMaximize,
        LpProblem,
        LpStatus,
        LpVariable,
        lpSum,
        value,
    )
except ImportError:
    print("Ошибка: требуется библиотека pulp. Установите: pip install pulp")
    sys.exit(1)


# ------------------------------------------------------------
# Глобальные параметры задачи
# ------------------------------------------------------------

# Стоимость захода в мельницу.
VISIT_COST = 100

# Стоимость героя.
# В этой версии число героев фиксируется заранее вне модели,
# поэтому итоговая стоимость героев потом считается как:
# heroes * HERO_COST
HERO_COST = 2500

# Награда за одну мельницу.
REWARD = 500

# В этой упрощённой версии решаем только первый день.
DAY = 1


# ------------------------------------------------------------
# Небольшая служебная функция:
# проверяем, поддерживает ли класс солвера определённый параметр.
# Это нужно, потому что разные wrapper'ы в PuLP принимают немного
# разные наборы аргументов.
# ------------------------------------------------------------
def supports_param(cls, name: str) -> bool:
    try:
        return name in inspect.signature(cls).parameters
    except Exception:
        return False


# ------------------------------------------------------------
# Контекстный менеджер:
# временно меняем рабочую директорию.
#
# Это полезно, потому что некоторые солверы могут создавать
# служебные файлы (.mps, .sol и т.п.) в текущей папке.
# Мы хотим, чтобы эти файлы складывались в output_dir.
# ------------------------------------------------------------
@contextmanager
def working_directory(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ------------------------------------------------------------
# Настройка логирования.
#
# Логи пишутся:
#   1) в консоль;
#   2) в output_dir/debug.log
# ------------------------------------------------------------
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("simple_day1_mip_solver")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "debug.log", encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Логирование инициализировано")
    logger.info("Файл лога: %s", output_dir / "debug.log")
    return logger


# ------------------------------------------------------------
# Небольшая фабрика солверов.
#
# В этой версии поддерживаются:
#   - Gurobi
#   - HiGHS
#   - CBC
#
# Можно явно попросить один из них,
# а можно выбрать "auto" — тогда берём первый доступный.
#
# Важно:
# Gurobi через PuLP обычно работает через класс pulp.GUROBI.
# ------------------------------------------------------------
class SolverFactory:
    @staticmethod
    def available_solvers() -> Dict[str, bool]:
        result = {}

        # ----------------------------------------------------
        # Gurobi
        # ----------------------------------------------------
        gurobi_cls = getattr(pulp, "GUROBI", None)
        if gurobi_cls is not None:
            try:
                result["gurobi"] = bool(gurobi_cls(msg=False).available())
            except Exception:
                result["gurobi"] = False
        else:
            result["gurobi"] = False

        # ----------------------------------------------------
        # HiGHS
        # ----------------------------------------------------
        highs_cls = getattr(pulp, "HiGHS", None)
        if highs_cls is not None:
            try:
                result["highs"] = bool(highs_cls(msg=False).available())
            except Exception:
                result["highs"] = False
        else:
            result["highs"] = False

        # ----------------------------------------------------
        # CBC
        # ----------------------------------------------------
        cbc_cls = getattr(pulp, "PULP_CBC_CMD", None)
        if cbc_cls is not None:
            try:
                result["cbc"] = bool(cbc_cls(msg=False).available())
            except Exception:
                result["cbc"] = False
        else:
            result["cbc"] = False

        return result

    @staticmethod
    def print_available(logger: logging.Logger):
        logger.info("=" * 60)
        logger.info("ДОСТУПНЫЕ СОЛВЕРЫ")
        logger.info("=" * 60)
        for name, ok in SolverFactory.available_solvers().items():
            logger.info("  %-8s | %s", name.upper(), "✓ Доступен" if ok else "✗ Недоступен")
        logger.info("=" * 60)

    @staticmethod
    def _configure_solver_files(solver, output_dir: Path, logger: logging.Logger):
        """
        Пытаемся попросить солвер складывать служебные файлы
        в output_dir и по возможности сохранять их.

        Не все wrapper'ы это поддерживают.
        Например, у Gurobi через Python API многие вещи идут
        через память, а не через временные файлы.
        """
        try:
            if hasattr(solver, "tmpDir"):
                solver.tmpDir = str(output_dir)
            if hasattr(solver, "keepFiles"):
                solver.keepFiles = True
            logger.info("Папка для файлов солвера: %s", output_dir)
        except Exception as exc:
            logger.warning("Не удалось настроить папку файлов солвера: %s", exc)

    @staticmethod
    def get_solver(name: str, time_limit: int, threads: int, output_dir: Path, logger: logging.Logger):
        name = name.lower()

        # ----------------------------------------------------
        # Gurobi
        #
        # Используем wrapper pulp.GUROBI.
        # Если Gurobi установлен правильно и лицензия доступна,
        # этот солвер обычно работает быстрее остальных.
        #
        # Параметры:
        #   - timeLimit / TimeLimit: лимит времени
        #   - Threads: число потоков
        #   - MIPGap: 0.0 — просим точное решение в пределах лимита
        #   - LogToConsole: вывод логов в консоль
        #   - LogFile: отдельный лог Gurobi, если поддерживается
        # ----------------------------------------------------
        if name == "gurobi":
            solver_cls = getattr(pulp, "GUROBI", None)
            if solver_cls is None:
                return None

            kwargs: Dict[str, Any] = {"msg": True}

            # Часть параметров у wrapper'а может быть формальными
            # аргументами конструктора, а часть — solverParams.
            if supports_param(solver_cls, "timeLimit"):
                kwargs["timeLimit"] = time_limit
            if supports_param(solver_cls, "warmStart"):
                kwargs["warmStart"] = False
            if supports_param(solver_cls, "logPath"):
                kwargs["logPath"] = str(output_dir / "gurobi.log")

            try:
                solver = solver_cls(
                    **kwargs,
                    Threads=threads,
                    TimeLimit=time_limit,
                    MIPGap=0.0,
                    LogToConsole=1,
                    LogFile=str(output_dir / "gurobi.log"),
                )
                SolverFactory._configure_solver_files(solver, output_dir, logger)
                logger.info("Инициализирован Gurobi | params=%s", kwargs)
                return solver
            except Exception as exc:
                logger.warning("Gurobi недоступен: %s", exc)
                return None

        # ----------------------------------------------------
        # HiGHS
        # ----------------------------------------------------
        if name == "highs":
            solver_cls = getattr(pulp, "HiGHS", None)
            if solver_cls is None:
                return None

            kwargs: Dict[str, Any] = {"msg": True}
            if supports_param(solver_cls, "timeLimit"):
                kwargs["timeLimit"] = time_limit
            if supports_param(solver_cls, "threads"):
                kwargs["threads"] = threads
            if supports_param(solver_cls, "warmStart"):
                kwargs["warmStart"] = False
            if supports_param(solver_cls, "keepFiles"):
                kwargs["keepFiles"] = True

            try:
                solver = solver_cls(**kwargs)
                SolverFactory._configure_solver_files(solver, output_dir, logger)
                logger.info("Инициализирован HiGHS | params=%s", kwargs)
                return solver
            except Exception as exc:
                logger.warning("HiGHS недоступен: %s", exc)
                return None

        # ----------------------------------------------------
        # CBC
        # ----------------------------------------------------
        if name == "cbc":
            solver_cls = getattr(pulp, "PULP_CBC_CMD", None)
            if solver_cls is None:
                return None

            kwargs: Dict[str, Any] = {"msg": True}
            if supports_param(solver_cls, "timeLimit"):
                kwargs["timeLimit"] = time_limit
            if supports_param(solver_cls, "threads"):
                kwargs["threads"] = threads
            if supports_param(solver_cls, "warmStart"):
                kwargs["warmStart"] = False
            if supports_param(solver_cls, "keepFiles"):
                kwargs["keepFiles"] = True

            try:
                solver = solver_cls(**kwargs)
                SolverFactory._configure_solver_files(solver, output_dir, logger)
                logger.info("Инициализирован CBC | params=%s", kwargs)
                return solver
            except Exception as exc:
                logger.warning("CBC недоступен: %s", exc)
                return None

        return None

    @staticmethod
    def get_best(time_limit: int, threads: int, output_dir: Path, logger: logging.Logger):
        """
        В режиме auto пробуем солверы по простому приоритету:
        сначала HiGHS, потом CBC, потом Gurobi.
        """
        for name in ["highs", "cbc", "gurobi"]:
            solver = SolverFactory.get_solver(name, time_limit, threads, output_dir, logger)
            if solver is not None:
                logger.info("Выбран солвер: %s", name.upper())
                return solver, name
        return None, None


# ------------------------------------------------------------
# Класс данных задачи для одного дня.
#
# Что он делает:
#   - читает CSV;
#   - оставляет только объекты первого дня;
#   - переиндексирует их во внутренние индексы 1..N;
#   - строит матрицу расстояний только для этих объектов;
#   - для каждого героя определяет, какие объекты вообще могут быть
#     первыми в маршруте.
#
# Важно:
# мы работаем во внутренней нумерации 1..N, потому что так удобнее
# строить математическую модель.
# ------------------------------------------------------------
class VRPData:
    def __init__(self, data_dir: str, num_heroes: int, logger: logging.Logger):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.num_heroes = num_heroes

        self.heroes_df: Optional[pd.DataFrame] = None
        self.objects_df: Optional[pd.DataFrame] = None

        # object_ids — реальные object_id из исходных данных
        self.object_ids: Optional[np.ndarray] = None
        self.object_count: int = 0

        # Преобразование:
        # внешний object_id -> внутренний индекс 1..N
        self.id_to_idx: Dict[int, int] = {}

        # И обратно:
        # внутренний индекс 1..N -> внешний object_id
        self.idx_to_id: Dict[int, int] = {}

        # Список всех внутренних индексов объектов дня.
        self.objects: List[int] = []

        # Расстояние от таверны до объекта:
        # dist_start_internal[internal_idx]
        self.dist_start_internal: Dict[int, int] = {}

        # Матрица расстояний между объектами дня.
        # Размер (N+1) x (N+1), индекс 0 не используется для объектов.
        self.dist_matrix_internal: Optional[np.ndarray] = None

        # starts_by_hero[k] = список объектов, с которых герой k
        # может начать маршрут с учётом "последнего хода".
        self.starts_by_hero: Dict[int, List[int]] = {}

        self._load()

    def _load(self):
        self.logger.info("Загрузка данных...")
        t0 = time.time()

        heroes_path = self.data_dir / "data_heroes.csv"
        objects_path = self.data_dir / "data_objects.csv"
        dist_start_path = self.data_dir / "dist_start.csv"
        dist_matrix_path = self.data_dir / "dist_objects.csv"

        # ----------------------------------------------------
        # Загружаем первых num_heroes героев.
        # ----------------------------------------------------
        self.heroes_df = (
            pd.read_csv(heroes_path)
            .sort_values("hero_id")
            .head(self.num_heroes)
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Берём только объекты дня DAY.
        # ----------------------------------------------------
        all_objects = pd.read_csv(objects_path).sort_values("object_id").reset_index(drop=True)
        self.objects_df = (
            all_objects[all_objects["day_open"] == DAY]
            .copy()
            .sort_values("object_id")
            .reset_index(drop=True)
        )

        self.object_ids = self.objects_df["object_id"].astype(int).to_numpy()
        self.object_count = len(self.object_ids)

        if self.object_count == 0:
            raise ValueError(f"Для дня {DAY} нет объектов")

        # ----------------------------------------------------
        # Строим переиндексацию.
        # ----------------------------------------------------
        for pos, obj_id in enumerate(self.object_ids, start=1):
            self.id_to_idx[int(obj_id)] = pos
            self.idx_to_id[pos] = int(obj_id)

        self.objects = list(range(1, self.object_count + 1))

        # ----------------------------------------------------
        # Загружаем dist_start и переводим во внутренние индексы.
        # ----------------------------------------------------
        dist_start_df = pd.read_csv(dist_start_path).sort_values("object_id").reset_index(drop=True)
        dist_start_map = {
            int(row.object_id): int(row.dist_start)
            for row in dist_start_df.itertuples(index=False)
        }

        for obj_id in self.object_ids:
            self.dist_start_internal[self.id_to_idx[int(obj_id)]] = dist_start_map[int(obj_id)]

        # ----------------------------------------------------
        # Загружаем полную матрицу расстояний и вырезаем из неё
        # только объекты нужного дня.
        # ----------------------------------------------------
        full_dist = pd.read_csv(dist_matrix_path).to_numpy(dtype=np.int32)
        source_idx = (self.object_ids - 1).astype(int)
        sub_dist = full_dist[np.ix_(source_idx, source_idx)]

        self.dist_matrix_internal = np.zeros((self.object_count + 1, self.object_count + 1), dtype=np.int32)
        self.dist_matrix_internal[1:, 1:] = sub_dist

        # ----------------------------------------------------
        # Для каждого героя заранее определяем допустимые стартовые объекты.
        #
        # Почему условие такое:
        #   dist_start(i) + 1 <= cap
        #
        # Потому что "последний ход" разрешает последнее посещение
        # за 1 очко, а не за 100.
        #
        # Значит, чтобы объект мог быть первым и одновременно последним
        # в маршруте, герою нужно:
        #   - дойти до него,
        #   - иметь хотя бы 1 очко на последний заход.
        # ----------------------------------------------------
        for k in range(self.num_heroes):
            cap = self.get_hero_capacity(k)
            starts = [i for i in self.objects if self.get_start_distance(i) + 1 <= cap]
            self.starts_by_hero[k] = starts

        self.logger.info("✓ Героев: %d", self.num_heroes)
        self.logger.info("✓ Объектов дня %d: %d", DAY, self.object_count)
        self.logger.info("✓ Время загрузки: %.2f сек", time.time() - t0)

    # --------------------------------------------------------
    # Вспомогательные методы доступа к данным
    # --------------------------------------------------------
    def get_distance(self, i: int, j: int) -> int:
        return int(self.dist_matrix_internal[i, j])

    def get_start_distance(self, i: int) -> int:
        return int(self.dist_start_internal[i])

    def get_hero_capacity(self, k: int) -> int:
        return int(self.heroes_df.iloc[k]["move_points"])

    def get_hero_id(self, k: int) -> int:
        return int(self.heroes_df.iloc[k]["hero_id"])


# ------------------------------------------------------------
# Главный класс MIP-модели.
#
# Здесь строится математическая модель маршрутизации.
#
# Основные переменные:
#
# a_k:
#   герой k активен или нет
#
# v_{i,k}:
#   объект i посещён героем k или нет
#
# x_{i,j,k}:
#   герой k идёт по дуге i -> j
#
# u_{i,k}:
#   служебная MTZ-переменная для устранения циклов
#
# Важно:
# число героев фиксировано ВНЕ модели.
# Это значит:
#   - стоимость героев в objective НЕ входит,
#   - потому что она постоянна и не влияет на оптимум.
#
# Поэтому внутри модели мы максимизируем:
#   число посещённых объектов
#   + маленький epsilon * leftover
#
# leftover нужен только как tie-breaker:
# если два решения покрывают одинаковое число объектов,
# предпочитаем то, где у героев осталось больше хода.
# ------------------------------------------------------------
class SimpleVRPMIPModel:
    def __init__(
        self,
        data: VRPData,
        time_limit: int,
        threads: int,
        solver_name: str,
        output_dir: Path,
        logger: logging.Logger,
    ):
        self.data = data
        self.time_limit = time_limit
        self.threads = threads
        self.solver_name = solver_name.lower()
        self.output_dir = output_dir
        self.logger = logger

        self.model: Optional[LpProblem] = None
        self.solver = None
        self.actual_solver_name = None

        # a_k: герой активен
        self.a_vars: Dict[int, LpVariable] = {}

        # v_{i,k}: герой k посещает объект i
        self.v_vars: Dict[Tuple[int, int], LpVariable] = {}

        # x_{i,j,k}: герой k идёт по дуге i -> j
        self.x_vars: Dict[Tuple[int, int, int], LpVariable] = {}

        # u_{i,k}: MTZ-переменная для борьбы с подциклами
        self.u_vars: Dict[Tuple[int, int], LpVariable] = {}

        # Список всех возможных дуг между разными объектами.
        self.arcs: List[Tuple[int, int]] = []

        # Маленький коэффициент для tie-breaker.
        self.tie_break_epsilon: float = 0.0

        self._build_model()

    def _build_model(self):
        self.logger.info("Построение упрощённой MIP-модели...")
        t0 = time.time()

        heroes = list(range(self.data.num_heroes))
        objects = self.data.objects
        n = self.data.object_count

        # ----------------------------------------------------
        # Все возможные дуги между объектами.
        # Дугу i -> i не создаём.
        # ----------------------------------------------------
        self.arcs = [(i, j) for i in objects for j in objects if i != j]

        # ----------------------------------------------------
        # Создаём задачу максимизации.
        # ----------------------------------------------------
        self.model = LpProblem("SimpleVRPDay1Model", LpMaximize)

        # ----------------------------------------------------
        # Создаём переменные.
        # ----------------------------------------------------
        for k in heroes:
            self.a_vars[k] = LpVariable(f"a_{k}", cat=LpBinary)

            for i in objects:
                self.v_vars[(i, k)] = LpVariable(f"v_{i}_{k}", cat=LpBinary)

                # MTZ-переменная.
                # Если объект не посещается, u может быть 0.
                # Если посещается, u показывает его порядок в маршруте.
                self.u_vars[(i, k)] = LpVariable(
                    f"u_{i}_{k}",
                    lowBound=0,
                    upBound=n,
                    cat=LpContinuous,
                )

            for i, j in self.arcs:
                self.x_vars[(i, j, k)] = LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)

        # ----------------------------------------------------
        # Здесь будем накапливать выражение для суммарного leftover.
        # Потом добавим его в objective с маленьким коэффициентом.
        # ----------------------------------------------------
        total_leftover_expr = 0

        # ----------------------------------------------------
        # Ограничения строим отдельно для каждого героя.
        # ----------------------------------------------------
        for k in heroes:
            cap = self.data.get_hero_capacity(k)
            starts_k = set(self.data.starts_by_hero[k])

            # ------------------------------------------------
            # Для удобства заранее строим:
            # in_expr[i]  = сумма всех дуг, входящих в i
            # out_expr[i] = сумма всех дуг, исходящих из i
            #
            # Всё это для конкретного героя k.
            # ------------------------------------------------
            in_expr: Dict[int, Any] = {}
            out_expr: Dict[int, Any] = {}

            for i in objects:
                in_expr[i] = lpSum(self.x_vars[(p, i, k)] for p in objects if p != i)
                out_expr[i] = lpSum(self.x_vars[(i, s, k)] for s in objects if s != i)

            # ------------------------------------------------
            # visit_count = сколько объектов посетил герой
            # start_count = сколько стартов у маршрута
            # end_count   = сколько концов у маршрута
            #
            # Почему start_count считается как:
            #   v(i,k) - in(i,k)
            #
            # Если объект посещён и в него нет входящей дуги,
            # то это старт маршрута.
            #
            # Аналогично для end_count:
            #   v(i,k) - out(i,k)
            # ------------------------------------------------
            visit_count = lpSum(self.v_vars[(i, k)] for i in objects)
            start_count = lpSum(self.v_vars[(i, k)] - in_expr[i] for i in objects)
            end_count = lpSum(self.v_vars[(i, k)] - out_expr[i] for i in objects)

            # ------------------------------------------------
            # Если герой активен, у него должен быть ровно один старт.
            # Если не активен — стартов 0.
            # ------------------------------------------------
            self.model += start_count == self.a_vars[k], f"StartCount_{k}"

            # ------------------------------------------------
            # Аналогично для конца маршрута:
            # у активного героя один конец, у неактивного 0.
            # ------------------------------------------------
            self.model += end_count == self.a_vars[k], f"EndCount_{k}"

            # ------------------------------------------------
            # Если герой не активен (a_k = 0),
            # то visit_count должен быть 0.
            #
            # Если активен, то visit_count может быть до n.
            # ------------------------------------------------
            self.model += visit_count <= n * self.a_vars[k], f"ActiveVisitLink_{k}"

            # ------------------------------------------------
            # Для каждого объекта:
            #
            # 1. входящая степень не может быть больше факта посещения;
            # 2. исходящая степень не может быть больше факта посещения.
            #
            # То есть если объект не посещён, никаких дуг через него быть не должно.
            # ------------------------------------------------
            for i in objects:
                self.model += in_expr[i] <= self.v_vars[(i, k)], f"InLeVisit_{i}_{k}"
                self.model += out_expr[i] <= self.v_vars[(i, k)], f"OutLeVisit_{i}_{k}"

                # --------------------------------------------
                # Если объект не может быть стартовым,
                # то если герой его посетил, у него должен быть вход.
                #
                # Это запрещает начинать маршрут в недопустимой точке.
                # --------------------------------------------
                if i not in starts_k:
                    self.model += self.v_vars[(i, k)] <= in_expr[i], f"NoStartHere_{i}_{k}"

                # --------------------------------------------
                # Ограничения на MTZ-переменные.
                #
                # Если объект не посещён:
                #   u может быть 0
                #
                # Если посещён:
                #   u >= 1
                #   u <= n
                # --------------------------------------------
                self.model += self.u_vars[(i, k)] >= self.v_vars[(i, k)], f"Ulb_{i}_{k}"
                self.model += self.u_vars[(i, k)] <= n * self.v_vars[(i, k)], f"Uub_{i}_{k}"

            # ------------------------------------------------
            # Считаем стоимость маршрута героя.
            #
            # start_cost:
            #   стоимость дороги от таверны до первого объекта.
            #
            # Почему формула:
            #   dist_start(i) * (v(i,k) - in(i,k))
            #
            # Потому что (v - in) = 1 только для стартовой вершины маршрута.
            # ------------------------------------------------
            start_cost = lpSum(
                self.data.get_start_distance(i) * (self.v_vars[(i, k)] - in_expr[i])
                for i in objects
            )

            # ------------------------------------------------
            # arc_cost:
            #   стоимость всех переходов между объектами маршрута.
            # ------------------------------------------------
            arc_cost = lpSum(
                self.data.get_distance(i, j) * self.x_vars[(i, j, k)]
                for i, j in self.arcs
            )

            # ------------------------------------------------
            # visit_cost:
            #   если герой посещает m объектов,
            #   базово это стоит 100 * m.
            # ------------------------------------------------
            visit_cost = VISIT_COST * visit_count

            # ------------------------------------------------
            # Правило последнего хода.
            #
            # В игре последнее посещение можно сделать не за 100,
            # а имея хотя бы 1 очко хода.
            #
            # Поэтому если герой активен, мы вычитаем 99:
            #   100 превращается в 1 для последнего посещения.
            #
            # Это работает потому, что у активного героя ровно один маршрут.
            # ------------------------------------------------
            last_move_correction = (VISIT_COST - 1) * self.a_vars[k]

            # ------------------------------------------------
            # used_cost — полная стоимость маршрута героя
            # в нашей упрощённой модели.
            # ------------------------------------------------
            used_cost = start_cost + arc_cost + visit_cost - last_move_correction

            # ------------------------------------------------
            # Ограничение по ёмкости героя:
            # маршрут не должен требовать больше хода,
            # чем доступно герою.
            # ------------------------------------------------
            self.model += (used_cost <= cap), f"Capacity_{k}"

            # ------------------------------------------------
            # leftover_k — сколько "запаса хода" осталось.
            #
            # Для неактивного героя:
            #   a_k = 0, used_cost = 0, leftover = 0
            #
            # Для активного:
            #   leftover = capacity - used_cost
            #
            # Это не главная цель, а только tie-breaker.
            # ------------------------------------------------
            leftover_k = cap * self.a_vars[k] - used_cost
            total_leftover_expr += leftover_k

            # ------------------------------------------------
            # MTZ-ограничения от подциклов.
            #
            # Идея:
            # если дуга i -> j используется,
            # то порядок u_j должен быть больше u_i.
            #
            # Это запрещает появление маленьких циклов,
            # не связанных со стартом.
            # ------------------------------------------------
            for i, j in self.arcs:
                self.model += (
                    self.u_vars[(i, k)] - self.u_vars[(j, k)] + n * self.x_vars[(i, j, k)] <= n - 1,
                    f"MTZ_{i}_{j}_{k}",
                )

        # ----------------------------------------------------
        # Глобальная уникальность посещений:
        # каждый объект может быть посещён не более одного раза
        # всеми героями вместе.
        #
        # Без этого ограничения MIP может назначать один и тот
        # же object_id нескольким героям, что запрещено
        # правилами задачи.
        # ----------------------------------------------------
        for i in objects:
            self.model += (
                lpSum(self.v_vars[(i, k)] for k in heroes) <= 1,
                f"UniqueVisit_{i}",
            )

        # ----------------------------------------------------
        # Целевая функция.
        #
        # Так как число героев фиксировано вне модели,
        # их стоимость — константа.
        #
        # Значит внутри модели достаточно максимизировать
        # число посещённых объектов.
        #
        # Но мы хотим tie-breaker как в LNS:
        #   сначала максимум посещённых,
        #   потом максимум leftover.
        #
        # Для этого добавляем очень маленький коэффициент epsilon:
        #   objective = visited + epsilon * leftover
        #
        # Важно:
        # epsilon должен быть настолько маленьким,
        # чтобы увеличение leftover никогда не перевесило
        # увеличение числа посещённых объектов на 1.
        # ----------------------------------------------------
        sum_caps = sum(self.data.get_hero_capacity(k) for k in heroes)

        # Берём очень маленький коэффициент.
        # Даже если leftover у всех героев суммарно большой,
        # эта добавка всё равно будет меньше 1.
        self.tie_break_epsilon = 1.0 / max(1, sum_caps + 1)

        self.model += (
            lpSum(self.v_vars.values()) + self.tie_break_epsilon * total_leftover_expr,
            "VisitedThenLeftover",
        )

        self.logger.info("✓ Tie-break epsilon: %.12f", self.tie_break_epsilon)
        self.logger.info("✓ Переменных: %d", len(self.model.variables()))
        self.logger.info("✓ Ограничений: %d", len(self.model.constraints))
        self.logger.info("✓ Модель построена за %.2f сек", time.time() - t0)

    def _init_solver(self) -> bool:
        """
        Инициализируем выбранный солвер.
        Если solver=auto, берём первый доступный.
        """
        if self.solver_name != "auto":
            self.solver = SolverFactory.get_solver(
                self.solver_name,
                self.time_limit,
                self.threads,
                self.output_dir,
                self.logger,
            )
            if self.solver is not None:
                self.actual_solver_name = self.solver_name
                return True

        self.solver, self.actual_solver_name = SolverFactory.get_best(
            self.time_limit,
            self.threads,
            self.output_dir,
            self.logger,
        )
        return self.solver is not None

    def solve(self) -> Tuple[bool, Optional[float], str]:
        """
        Запускаем решение модели.
        Возвращаем:
          - найдено ли решение;
          - значение objective;
          - статус солвера.
        """
        if not self._init_solver():
            return False, None, "NoSolver"

        self.logger.info("=" * 60)
        self.logger.info("ЗАПУСК РЕШЕНИЯ")
        self.logger.info("=" * 60)
        self.logger.info("Лимит времени: %d сек", self.time_limit)
        self.logger.info("Потоков: %d", self.threads)
        self.logger.info("Солвер: %s", self.actual_solver_name)

        t0 = time.time()

        # Работаем из output_dir, чтобы служебные файлы солвера
        # складывались туда.
        with working_directory(self.output_dir):
            status_code = self.model.solve(self.solver)

        elapsed = time.time() - t0

        status_str = LpStatus.get(status_code, str(status_code))
        obj_value = value(self.model.objective)

        self.logger.info("✓ Время решения: %.2f сек", elapsed)
        self.logger.info("✓ Статус: %s", status_str)
        self.logger.info("✓ Objective: %s", obj_value)

        has_solution = obj_value is not None and status_str not in {"Infeasible", "Unbounded", "Undefined"}
        return has_solution, obj_value, status_str

    def extract_routes(self) -> Dict[int, List[int]]:
        """
        Извлекаем маршруты из решения MIP.

        Как это работает:
        1. Смотрим, какие вершины посещены по v(i,k).
        2. Смотрим, какие дуги выбраны по x(i,j,k).
        3. Находим стартовую вершину — у неё входящая степень 0.
        4. Идём по succ, пока маршрут не закончится.
        """
        self.logger.info("Извлечение маршрутов...")
        routes: Dict[int, List[int]] = {}

        for k in range(self.data.num_heroes):
            hero_id = self.data.get_hero_id(k)

            visited = [
                i for i in self.data.objects
                if value(self.v_vars[(i, k)]) is not None and value(self.v_vars[(i, k)]) > 0.5
            ]

            if not visited:
                routes[hero_id] = []
                continue

            succ: Dict[int, int] = {}
            indeg: Dict[int, int] = {i: 0 for i in visited}

            for i, j in self.arcs:
                xv = value(self.x_vars[(i, j, k)])
                if xv is not None and xv > 0.5:
                    if i in succ:
                        raise RuntimeError(f"У героя {hero_id} больше одной исходящей дуги")
                    succ[i] = j
                    if j in indeg:
                        indeg[j] += 1

            starts = [i for i in visited if indeg.get(i, 0) == 0]
            if len(starts) != 1:
                raise RuntimeError(f"У героя {hero_id} найдено {len(starts)} стартов, ожидался 1")

            cur = starts[0]
            seen = set()
            route_internal = []

            while True:
                if cur in seen:
                    raise RuntimeError(f"У героя {hero_id} обнаружен цикл при извлечении маршрута")
                seen.add(cur)
                route_internal.append(cur)
                if cur not in succ:
                    break
                cur = succ[cur]

            if len(route_internal) != len(visited):
                raise RuntimeError(
                    f"У героя {hero_id}: длина маршрута {len(route_internal)}, а visited={len(visited)}"
                )

            routes[hero_id] = [self.data.idx_to_id[i] for i in route_internal]

        return routes

    def route_cost(self, route: List[int]) -> int:
        """
        Считаем стоимость маршрута уже во внешних object_id.
        Используется для проверки результата.

        Формула та же:
          start_distance(first) + 1
          + sum(dist(a,b) + 100)
        """
        if not route:
            return 0

        first_idx = self.data.id_to_idx[route[0]]
        total = self.data.get_start_distance(first_idx) + 1

        for a, b in zip(route[:-1], route[1:]):
            ia = self.data.id_to_idx[a]
            ib = self.data.id_to_idx[b]
            total += self.data.get_distance(ia, ib) + VISIT_COST

        return int(total)

    def validate_routes(self, routes: Dict[int, List[int]]) -> bool:
        """
        Базовая проверка маршрутов после извлечения.

        Проверяем:
          - нет ли повторов объектов;
          - не посещён ли объект двумя героями;
          - допустим ли стартовый объект;
          - не превышена ли ёмкость героя.
        """
        self.logger.info("Проверка маршрутов...")
        ok = True
        visited_global = set()

        for k in range(self.data.num_heroes):
            hero_id = self.data.get_hero_id(k)
            route = routes.get(hero_id, [])

            if len(route) != len(set(route)):
                self.logger.error("  ✗ Герой %d: повтор объекта в маршруте", hero_id)
                ok = False

            for obj_id in route:
                if obj_id in visited_global:
                    self.logger.error("  ✗ Объект %d посещён больше одного раза", obj_id)
                    ok = False
                visited_global.add(obj_id)

            if route:
                first_idx = self.data.id_to_idx[route[0]]
                if first_idx not in set(self.data.starts_by_hero[k]):
                    self.logger.error("  ✗ Герой %d: первый объект %d не может быть стартом", hero_id, route[0])
                    ok = False

            used = self.route_cost(route)
            cap = self.data.get_hero_capacity(k)
            if used > cap:
                self.logger.error("  ✗ Герой %d: переполнение %d/%d", hero_id, used, cap)
                ok = False
            else:
                if route:
                    self.logger.info("  ✓ Герой %d: %d/%d", hero_id, used, cap)

        if ok:
            self.logger.info("✓ Все маршруты валидны")
        return ok


def main():
    """
    Главная функция:
      1. читаем аргументы;
      2. проверяем данные;
      3. строим модель;
      4. решаем её;
      5. извлекаем маршруты;
      6. сохраняем результаты.
    """
    parser = argparse.ArgumentParser(
        description="Упрощённый MIP solver для первого дня VRP (Data Fusion 2026 - Heroes Task)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-dir", type=str, default="data/", help="Папка с данными")
    parser.add_argument("--output-dir", type=str, default="out_mip/", help="Папка для результатов")
    parser.add_argument("--heroes", type=int, default=17, help="Сколько первых героев использовать")
    parser.add_argument("--time-limit", type=int, default=10000, help="Лимит времени, сек")
    parser.add_argument("--threads", type=int, default=4, help="Число потоков")
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        choices=["auto", "gurobi", "highs", "cbc"],
        help="Какой MIP-солвер использовать",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir, args.log_level)

    if args.heroes <= 0:
        logger.error("--heroes должно быть > 0")
        return 1

    logger.info("=" * 80)
    logger.info("УПРОЩЁННЫЙ DAY 1 MIP SOLVER")
    logger.info("=" * 80)
    logger.info("Данные: %s", args.data_dir)
    logger.info("Выходная папка: %s", output_dir)
    logger.info("Героев (фиксировано вне модели): %d", args.heroes)
    logger.info("День: %d", DAY)
    logger.info("Лимит времени: %d сек", args.time_limit)
    logger.info("Потоков: %d", args.threads)
    logger.info("Солвер: %s", args.solver.upper())
    logger.info("=" * 80)

    SolverFactory.print_available(logger)

    # --------------------------------------------------------
    # Проверяем наличие всех нужных файлов.
    # --------------------------------------------------------
    required_files = ["data_heroes.csv", "data_objects.csv", "dist_start.csv", "dist_objects.csv"]
    for name in required_files:
        path = Path(args.data_dir) / name
        if not path.exists():
            logger.error("Файл не найден: %s", path)
            return 1
        logger.info("✓ Файл найден: %s", name)

    # --------------------------------------------------------
    # Загружаем данные.
    # --------------------------------------------------------
    try:
        data = VRPData(args.data_dir, args.heroes, logger)
    except Exception as exc:
        logger.exception("Ошибка загрузки данных: %s", exc)
        return 1

    # --------------------------------------------------------
    # Строим и решаем модель.
    # --------------------------------------------------------
    try:
        model = SimpleVRPMIPModel(
            data=data,
            time_limit=args.time_limit,
            threads=args.threads,
            solver_name=args.solver,
            output_dir=output_dir,
            logger=logger,
        )
        success, obj_value, status_str = model.solve()
    except Exception as exc:
        logger.exception("Ошибка построения или решения модели: %s", exc)
        return 1

    # --------------------------------------------------------
    # Извлекаем маршруты.
    # --------------------------------------------------------
    try:
        if obj_value is not None:
            routes = model.extract_routes()
            is_valid = model.validate_routes(routes)
            if not is_valid:
                logger.error("Маршруты невалидны. submission.csv не будет сохранён.")
                return 1
        else:
            logger.warning("Решение отсутствует, будут записаны пустые маршруты")
            routes = {data.get_hero_id(k): [] for k in range(data.num_heroes)}
    except Exception as exc:
        logger.exception("Ошибка извлечения маршрутов: %s", exc)
        return 1

    # --------------------------------------------------------
    # Так как число героев фиксировано вне модели,
    # стоимость героев — просто константа.
    # --------------------------------------------------------
    visited_count = sum(len(route) for route in routes.values())
    fixed_hero_cost = args.heroes * HERO_COST
    total_reward = visited_count * REWARD
    net_score = total_reward - fixed_hero_cost

    # --------------------------------------------------------
    # Сохраняем результат в JSON.
    # --------------------------------------------------------
    results = {
        "metadata": {
            "day": DAY,
            "heroes_fixed": args.heroes,
            "solver_requested": args.solver,
            "solver_used": model.actual_solver_name,
            "status": status_str,
            "objective_value": obj_value,
            "time_limit": args.time_limit,
            "threads": args.threads,
            "timestamp": datetime.now().isoformat(),
            "variables": len(model.model.variables()) if model.model is not None else None,
            "constraints": len(model.model.constraints) if model.model is not None else None,
            "tie_break_epsilon": model.tie_break_epsilon,
            "hero_cost_mode": "fixed_outside_model",
        },
        "routes": routes,
        "score": {
            "objects_visited": visited_count,
            "total_reward": total_reward,
            "hero_cost": fixed_hero_cost,
            "net_score": net_score,
        },
    }

    results_json = output_dir / "results.json"
    submissions_csv = output_dir / "submissions.csv"

    try:
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("✓ JSON сохранён: %s", results_json)
    except Exception as exc:
        logger.exception("Ошибка сохранения results.json: %s", exc)
        return 1

    # --------------------------------------------------------
    # Сохраняем submission в CSV-формате соревнования.
    # --------------------------------------------------------
    try:
        with open(submissions_csv, "w", encoding="utf-8") as f:
            f.write("hero_id,object_id\n")
            for hero_id, route in routes.items():
                for obj_id in route:
                    f.write(f"{hero_id},{obj_id}\n")
        logger.info("✓ CSV сохранён: %s", submissions_csv)
    except Exception as exc:
        logger.exception("Ошибка сохранения submissions.csv: %s", exc)
        return 1

    # --------------------------------------------------------
    # Краткий итог.
    # --------------------------------------------------------
    logger.info("=" * 80)
    logger.info("ИТОГИ")
    logger.info("=" * 80)
    logger.info("Статус: %s", status_str)
    logger.info("Посещено объектов: %d", visited_count)
    logger.info("Награда: %d", total_reward)
    logger.info("Стоимость героев (фиксированная): %d", fixed_hero_cost)
    logger.info("Итоговый счёт: %d", net_score)
    logger.info("Переменных: %d", len(model.model.variables()) if model.model is not None else 0)
    logger.info("Ограничений: %d", len(model.model.constraints) if model.model is not None else 0)
    logger.info("=" * 80)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
