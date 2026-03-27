# Задача «Герои» (Data Fusion 2026) — LNS и окружение

Папка собрана для публикации на GitHub: здесь **C++ LNS**, **Python LNS** из репозитория соревнования, наш **pipeline** на чистом Python и **базовый OR-Tools** (`task_3_ortools_reference.py`).

## Структура

| Путь | Содержимое |
|------|------------|
| `external/data-fusion-2026-heroes/` | Исходники соревнования: `lns_solver.cpp`, `lns_solver.py`, `mip_solver_day1.py`, `view_solution.py`, `README.md`, `LICENSE`, `data/` |
| `pipeline.py` | Локальный пайплайн LNS на Python (несколько сидов, `output/`) |
| `task_3_ortools_reference.py` | Вариант на OR-Tools (из корня проекта; для сравнения) |

Данные для запуска: папка **`data/`** в корне `task_3/` (те же четыре CSV, что и в основном проекте).  
В `external/data-fusion-2026-heroes/data/` может лежать заглушка из архива — для воспроизводимости используй `task_3/data/` или скопируй CSV соревнования туда вручную.

## Сборка C++ (MinGW / g++)

Из каталога с исходником:

```bash
cd external/data-fusion-2026-heroes
g++ -O3 -std=c++20 -o lns_solver.exe lns_solver.cpp
```

Запуск (пути **без кириллицы**, иначе возможна ошибка `Illegal byte sequence`):

```bash
lns_solver.exe --data-dir "C:/path/to/data" --output-dir "C:/path/to/out" --heroes 17 --seed 42 --day-time-limits 300,300,300,300,300,300,300
```

Список опций:

```bash
lns_solver.exe --help
```

## Python LNS (из external)

```bash
cd external/data-fusion-2026-heroes
pip install numpy pandas
python lns_solver.py
```

## Наш `pipeline.py`

Из каталога **`task_3`** (по умолчанию читает `./data` и пишет в `./output`):

```bash
cd task_3
pip install numpy pandas
python pipeline.py --fast
```

Пример с явными путями:

```bash
python pipeline.py --data-dir data --output-dir output --heroes-range 17,17 --seeds 42
```

См. `python pipeline.py --help`.

## Лицензия

Файлы в `external/data-fusion-2026-heroes/` распространяются на условиях `LICENSE` внутри этой папки. Дополнительные файлы в корне `task_3/` добавлены для удобства сборки репозитория.
