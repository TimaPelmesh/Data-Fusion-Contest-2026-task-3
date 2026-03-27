/*
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

    Компиляция:
      g++ -O2 -std=c++20 lns_solver.cpp -o lns_solver

    Пример запуска:
      ./lns_solver \
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
*/
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// Глобальные константы задачи
// ============================================================

// Стоимость посещения мельницы в очках хода.
static constexpr int VISIT_COST = 100;

// Стоимость найма одного героя.
static constexpr int HERO_COST = 2500;

// Количество дней в неделе.
static constexpr int DAYS = 7;

// ============================================================
// Вспомогательные функции
// ============================================================

// Текущее монотонное время в секундах.
// Используется только для контроля лимита времени алгоритма.
double now_sec() {
    using clock = std::chrono::steady_clock;
    auto tp = clock::now().time_since_epoch();
    return std::chrono::duration<double>(tp).count();
}

// Текущее время в виде строки.
// Нужно только для красивых логов.
std::string wall_timestamp() {
    auto t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Простой вывод строки в лог.
void log_msg(const std::string& msg) {
    std::cout << wall_timestamp() << " | " << msg << "\n";
}

// Удаляем пробелы в начале и конце строки.
std::string trim(const std::string& s) {
    size_t l = 0;
    while (l < s.size() && std::isspace(static_cast<unsigned char>(s[l]))) ++l;
    size_t r = s.size();
    while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) --r;
    return s.substr(l, r - l);
}

// Очень простой split по запятой.
// Для данной задачи этого достаточно, потому что все входные файлы
// представляют собой простые числовые таблицы.
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, ',')) {
        parts.push_back(item);
    }

    return parts;
}

// Парсим лимиты времени по дням.
// Можно дать одно число - тогда оно применяется ко всем дням.
// Или 7 чисел через запятую - отдельно на каждый день.
std::array<double, DAYS + 1> parse_day_time_limits(const std::string& s) {
    std::array<double, DAYS + 1> out{};
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        parts.push_back(trim(item));
    }

    if (parts.size() == 1) {
        double x = std::stod(parts[0]);
        for (int d = 1; d <= DAYS; ++d) out[d] = x;
        return out;
    }

    if (parts.size() != DAYS) {
        throw std::runtime_error("--day-time-limits должен содержать либо 1 число, либо 7 чисел");
    }

    for (int d = 1; d <= DAYS; ++d) {
        out[d] = std::stod(parts[d - 1]);
    }

    return out;
}

// Подсказка по использованию программы.
void print_usage() {
    std::cout
        << "Usage:\n"
        << "  lns_solver --data-dir PATH --output-dir PATH [options]\n\n"
        << "Options:\n"
        << "  --heroes N\n"
        << "  --day-time-limits X or x1,x2,x3,x4,x5,x6,x7\n"
        << "  --seed N\n"
        << "  --iterations N   (0 = until time limit)\n"
        << "  --rcl-size N\n"
        << "  --destroy-frac-min X\n"
        << "  --destroy-frac-max X\n"
        << "  --temp-start X\n"
        << "  --temp-end X\n"
        << "  --log-every N\n";
}

// ============================================================
// Конфигурация запуска
// ============================================================

struct Config {
    // Папка, где лежат входные данные.
    fs::path data_dir = "data";

    // Папка, куда сохраняем результаты работы.
    fs::path output_dir = "out_lns";

    // Сколько первых героев использовать.
    // В этой версии считаем, что парк героев фиксирован заранее.
    int heroes = 17;

    // Начальное зерно генератора случайных чисел.
    // От него зависит воспроизводимость результата.
    int seed = 42;

    // Максимум итераций LNS на один день.
    // Если 0, работаем до исчерпания времени.
    int iterations = 0;

    // Размер restricted candidate list в destroy_worst.
    // Мы сортируем кандидатов по "выгоде удаления",
    // а потом случайно выбираем одного из лучших rcl_size кандидатов.
    int rcl_size = 5;

    // Минимальная доля мельниц, которую можно удалить на фазе destroy.
    double destroy_frac_min = 0.10;

    // Максимальная доля мельниц, которую можно удалить на фазе destroy.
    double destroy_frac_max = 0.35;

    // Начальная температура simulated annealing.
    // Большая температура => чаще принимаем ухудшения в начале.
    double temp_start = 0.20;

    // Конечная температура simulated annealing.
    // Маленькая температура => в конце поиск становится почти жадным.
    double temp_end = 0.001;

    // Как часто печатать лог по итерациям.
    int log_every = 100;

    // Лимит времени для каждого дня.
    // day_time_limits[1] ... day_time_limits[7]
    std::array<double, DAYS + 1> day_time_limits{};
};

// Ручной парсер аргументов.
bool parse_args(int argc, char** argv, Config& cfg) {
    std::string day_time_limits_str = "300,30,30,30,30,30,30";

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];

        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Для параметра " + name + " не указано значение");
            }
            return argv[++i];
        };

        if (key == "--data-dir") {
            cfg.data_dir = fs::path(need_value(key));
        } else if (key == "--output-dir") {
            cfg.output_dir = fs::path(need_value(key));
        } else if (key == "--heroes") {
            cfg.heroes = std::stoi(need_value(key));
        } else if (key == "--day-time-limits") {
            day_time_limits_str = need_value(key);
        } else if (key == "--seed") {
            cfg.seed = std::stoi(need_value(key));
        } else if (key == "--iterations") {
            cfg.iterations = std::stoi(need_value(key));
        } else if (key == "--rcl-size") {
            cfg.rcl_size = std::stoi(need_value(key));
        } else if (key == "--destroy-frac-min") {
            cfg.destroy_frac_min = std::stod(need_value(key));
        } else if (key == "--destroy-frac-max") {
            cfg.destroy_frac_max = std::stod(need_value(key));
        } else if (key == "--temp-start") {
            cfg.temp_start = std::stod(need_value(key));
        } else if (key == "--temp-end") {
            cfg.temp_end = std::stod(need_value(key));
        } else if (key == "--log-every") {
            cfg.log_every = std::stoi(need_value(key));
        } else if (key == "--help" || key == "-h") {
            print_usage();
            return false;
        } else {
            throw std::runtime_error("Неизвестный параметр: " + key);
        }
    }

    if (cfg.data_dir.empty() || cfg.output_dir.empty()) {
        print_usage();
        throw std::runtime_error("Нужно указать --data-dir и --output-dir");
    }

    cfg.data_dir = fs::absolute(cfg.data_dir);
    cfg.output_dir = fs::absolute(cfg.output_dir);
    cfg.day_time_limits = parse_day_time_limits(day_time_limits_str);

    if (cfg.heroes <= 0) {
        throw std::runtime_error("--heroes должно быть > 0");
    }
    if (cfg.rcl_size <= 0) {
        throw std::runtime_error("--rcl-size должно быть > 0");
    }
    if (cfg.destroy_frac_min <= 0 || cfg.destroy_frac_max <= 0 || cfg.destroy_frac_min > cfg.destroy_frac_max) {
        throw std::runtime_error("Некорректные destroy-frac параметры");
    }
    if (cfg.log_every <= 0) {
        throw std::runtime_error("--log-every должно быть > 0");
    }

    return true;
}

// ============================================================
// Полные данные задачи
// ============================================================

struct HeroState {
    // Возле какой мельницы герой закончил предыдущий день.
    // 0 означает, что в нашей модели он стартует из таверны.
    int anchor_ext = 0;

    // "Скидка" на стартовый переход следующего дня.
    // Это упрощённый способ переноса неиспользованного движения между днями
    // в day-by-day модели.
    int carry_discount = 0;
};

struct FullData {
    // Дневной запас очков хода для героев.
    // hero_id отдельно не храним - считаем, что используем героев 1..K.
    std::vector<int> hero_caps;

    // Число мельниц в исходной задаче.
    int full_object_count = 0;

    // object_id -> день открытия объекта.
    std::vector<int> object_day_open;

    // Расстояние от таверны до мельницы.
    std::vector<int> dist_start_by_objid;

    // Полная матрица расстояний между мельницами.
    // Храним в одном векторе ради компактности.
    std::vector<int> dist_full_flat;

    int hero_count() const {
        return (int)hero_caps.size();
    }

    int dist_by_objid(int obj_a, int obj_b) const {
        return dist_full_flat[(obj_a - 1) * full_object_count + (obj_b - 1)];
    }

    static FullData load(const fs::path& data_dir) {
        FullData full;

        const fs::path heroes_path = data_dir / "data_heroes.csv";
        const fs::path objects_path = data_dir / "data_objects.csv";
        const fs::path dist_start_path = data_dir / "dist_start.csv";
        const fs::path dist_matrix_path = data_dir / "dist_objects.csv";

        log_msg("Загрузка данных из: " + data_dir.string());

        // ----------------------------------------------------
        // 1. Герои
        // ----------------------------------------------------
        {
            std::ifstream fin(heroes_path);
            if (!fin) {
                throw std::runtime_error("Не удалось открыть файл: " + heroes_path.string());
            }

            std::string line;
            std::getline(fin, line); // header

            std::vector<std::pair<int, int>> heroes;
            while (std::getline(fin, line)) {
                if (line.empty()) continue;
                auto parts = split_csv_line(line);
                if (parts.size() < 2) continue;

                int hero_id = std::stoi(parts[0]);
                int move_points = std::stoi(parts[1]);
                heroes.push_back({hero_id, move_points});
            }

            std::sort(heroes.begin(), heroes.end());

            full.hero_caps.reserve(heroes.size());
            for (const auto& [hero_id, move_points] : heroes) {
                (void)hero_id;
                full.hero_caps.push_back(move_points);
            }
        }

        // ----------------------------------------------------
        // 2. Мельницы и дни открытия
        // ----------------------------------------------------
        {
            std::ifstream fin(objects_path);
            if (!fin) {
                throw std::runtime_error("Не удалось открыть файл: " + objects_path.string());
            }

            std::string line;
            std::getline(fin, line); // header

            int max_obj_id = 0;
            std::vector<std::pair<int, int>> obj_day;

            while (std::getline(fin, line)) {
                if (line.empty()) continue;
                auto parts = split_csv_line(line);
                if (parts.size() < 3) continue;

                int object_id = std::stoi(parts[0]);
                int day_open = std::stoi(parts[1]);

                obj_day.push_back({object_id, day_open});
                max_obj_id = std::max(max_obj_id, object_id);
            }

            full.full_object_count = max_obj_id;
            full.object_day_open.assign(max_obj_id + 1, 0);

            for (const auto& [object_id, day_open] : obj_day) {
                full.object_day_open[object_id] = day_open;
            }
        }

        // ----------------------------------------------------
        // 3. Расстояния от таверны
        // ----------------------------------------------------
        {
            std::ifstream fin(dist_start_path);
            if (!fin) {
                throw std::runtime_error("Не удалось открыть файл: " + dist_start_path.string());
            }

            std::string line;
            std::getline(fin, line); // header

            full.dist_start_by_objid.assign(full.full_object_count + 1, 0);

            while (std::getline(fin, line)) {
                if (line.empty()) continue;
                auto parts = split_csv_line(line);
                if (parts.size() < 2) continue;

                int object_id = std::stoi(parts[0]);
                int dist_start = std::stoi(parts[1]);

                if (object_id >= 0 && object_id < (int)full.dist_start_by_objid.size()) {
                    full.dist_start_by_objid[object_id] = dist_start;
                }
            }
        }

        // ----------------------------------------------------
        // 4. Матрица расстояний между мельницами
        // ----------------------------------------------------
        {
            std::ifstream fin(dist_matrix_path);
            if (!fin) {
                throw std::runtime_error("Не удалось открыть файл: " + dist_matrix_path.string());
            }

            std::string line;
            if (!std::getline(fin, line)) {
                throw std::runtime_error("Пустой файл: " + dist_matrix_path.string());
            }

            std::vector<std::vector<int>> rows;
            rows.reserve(full.full_object_count);

            while (std::getline(fin, line)) {
                if (line.empty()) continue;
                auto parts = split_csv_line(line);

                std::vector<int> row;
                row.reserve(parts.size());
                for (const auto& s : parts) {
                    row.push_back(std::stoi(s));
                }
                rows.push_back(std::move(row));
            }

            if ((int)rows.size() != full.full_object_count) {
                throw std::runtime_error("Неверное число строк в " + dist_matrix_path.string());
            }

            full.dist_full_flat.assign(full.full_object_count * full.full_object_count, 0);

            for (int i = 0; i < full.full_object_count; ++i) {
                if ((int)rows[i].size() != full.full_object_count) {
                    throw std::runtime_error("Неверный размер матрицы в " + dist_matrix_path.string());
                }

                for (int j = 0; j < full.full_object_count; ++j) {
                    full.dist_full_flat[i * full.full_object_count + j] = rows[i][j];
                }
            }
        }

        log_msg(std::format("Героев: {}", full.hero_count()));
        log_msg(std::format("Мельниц: {}", full.full_object_count));
        return full;
    }
};

// ============================================================
// Данные одного дня
// ============================================================

// DayData - это посуточный "срез" задачи.
//
// Мы решаем не сразу всю неделю одним огромным маршрутом,
// а по дням: отдельно день 1, отдельно день 2 и т.д.
//
// Для этого на каждый день строим компактную подзадачу:
// - берём только мельницы, которые открываются в этот день;
// - строим расстояния между ними;
// - считаем стоимость старта каждого героя к каждой такой мельнице.
struct DayData {
    int day = 1;
    int num_heroes = 0;
    int object_count = 0;

    // Запас хода каждого героя в этот день.
    std::vector<int> hero_caps;

    // Перевод внутреннего индекса мельницы во внешний object_id.
    std::vector<int> object_ids_ext;

    // Стоимость "поставить мельницу первой" для героя.
    std::vector<int> start_cost_flat;

    // Матрица расстояний между мельницами этого дня.
    std::vector<int> dist_flat;

    int dist(int a, int b) const {
        return dist_flat[a * object_count + b];
    }

    int start_cost(int hero_idx, int obj_idx) const {
        return start_cost_flat[hero_idx * object_count + obj_idx];
    }

    int hero_capacity(int h) const {
        return hero_caps[h];
    }

    int object_id(int obj) const {
        return object_ids_ext[obj];
    }

    int route_cost(int hero_idx, std::span<const int> route) const {
        if (route.empty()) return 0;

        int total = start_cost(hero_idx, route[0]);
        for (int i = 0; i + 1 < (int)route.size(); ++i) {
            total += dist(route[i], route[i + 1]) + VISIT_COST;
        }
        return total;
    }

    // --------------------------------------------------------
    // build_for_day
    // --------------------------------------------------------
    //
    // Эта функция строит "локальную" задачу для одного дня.
    //
    // Что именно она делает:
    //
    // 1. Берёт только первых heroes_count героев.
    // 2. Оставляет только те мельницы, которые открываются в день day.
    // 3. Переиндексирует эти мельницы во внутренние индексы 0..object_count-1.
    // 4. Строит маленькую матрицу расстояний между мельницами этого дня.
    // 5. Для каждого героя и каждой мельницы считает стоимость старта:
    //      - если это 1-й день, стартуем из таверны;
    //      - иначе стартуем от последней мельницы (в маршруте) предыдущего дня;
    //      - учитываем carry_discount, то есть накопленный "запас" движения.
    //
    // Зачем всё это нужно:
    //
    // LNS на 700 объектах недели сразу был бы слишком тяжёлым и сложным.
    // Посуточная декомпозиция уменьшает размер задачи:
    // в конкретный день мы смотрим только на мельницы этого дня.
    //
    // Важная идея start_cost:
    //
    // Для первой мельницы маршрута мы не хотим каждый раз отдельно помнить,
    // откуда пришёл герой и сколько движения у него осталось.
    // Поэтому заранее считаем "стоимость поставить мельницу первой" (в маршрут).
    // Это упрощает дальнейшие операции вставки/удаления.
    static DayData build_for_day(
        const FullData& full,
        int day,
        int heroes_count,
        const std::vector<HeroState>& hero_states_before_day
    ) {
        DayData data;
        data.day = day;
        data.num_heroes = heroes_count;

        if (heroes_count > full.hero_count()) {
            throw std::runtime_error("Запрошено героев больше, чем доступно");
        }
        if ((int)hero_states_before_day.size() != heroes_count) {
            throw std::runtime_error("hero_states_before_day.size() != heroes_count");
        }

        // Берём только первых heroes_count героев.
        data.hero_caps.reserve(heroes_count);
        for (int h = 0; h < heroes_count; ++h) {
            data.hero_caps.push_back(full.hero_caps[h]);
        }

        // Выбираем только мельницы, которые открываются именно сегодня.
        // Это и есть "дневной набор задач" для LNS.
        for (int obj_id = 1; obj_id <= full.full_object_count; ++obj_id) {
            if (obj_id < (int)full.object_day_open.size() && full.object_day_open[obj_id] == day) {
                data.object_ids_ext.push_back(obj_id);
            }
        }

        data.object_count = (int)data.object_ids_ext.size();

        // Строим маленькую матрицу расстояний только между мельницами дня.
        // Это уменьшает объём данных, с которыми работает поиск.
        data.dist_flat.assign(data.object_count * data.object_count, 0);
        for (int i = 0; i < data.object_count; ++i) {
            int obj_i = data.object_ids_ext[i];
            for (int j = 0; j < data.object_count; ++j) {
                int obj_j = data.object_ids_ext[j];
                data.dist_flat[i * data.object_count + j] = full.dist_by_objid(obj_i, obj_j);
            }
        }

        // start_cost_flat[h, j]:
        // стоимость поставить мельницу j первой в маршруте героя h.
        data.start_cost_flat.assign(data.num_heroes * data.object_count, 0);

        for (int h = 0; h < data.num_heroes; ++h) {
            const auto& hs = hero_states_before_day[h];

            for (int j = 0; j < data.object_count; ++j) {
                int obj_ext = data.object_ids_ext[j];

                int base_dist = 0;
                int carry = 0;

                if (day == 1) {
                    // В первый день герой стартует из таверны.
                    base_dist = full.dist_start_by_objid[obj_ext];
                    carry = 0;
                } else {
                    // В следующие дни стартуем от точки завершения предыдущего дня.
                    if (hs.anchor_ext == 0) {
                        base_dist = full.dist_start_by_objid[obj_ext];
                    } else {
                        base_dist = full.dist_by_objid(hs.anchor_ext, obj_ext);
                    }
                    carry = hs.carry_discount;
                }

                // Если накопленной "скидки" хватает на дорогу,
                // остаётся только VISIT_COST.
                if (carry >= base_dist) {
                    data.start_cost_flat[h * data.object_count + j] = VISIT_COST;
                } else {
                    data.start_cost_flat[h * data.object_count + j] = (base_dist - carry) + VISIT_COST;
                }
            }
        }

        return data;
    }
};

// ============================================================
// Решение одного дня
// ============================================================

struct Solution {
    using Route = std::vector<int>;

    // Ссылка на данные конкретного дня.
    // Через неё можно быстро получать расстояния и стартовые стоимости.
    const DayData* data = nullptr;

    // routes[r] - маршрут героя r.
    // Внутри маршрута лежат внутренние индексы мельниц дня.
    // Например, если routes[2] = {5, 1, 7}, значит герой 2 посещает сначала
    // мельницу №5 (во внутренней нумерации дня), потом №1, потом №7.
    std::vector<Route> routes;

    // route_costs[r] - стоимость маршрута routes[r].
    // Мы кэшируем эти значения, чтобы не пересчитывать стоимость маршрута целиком
    // после каждой маленькой операции.
    std::vector<int> route_costs;

    // obj_route[obj] - номер маршрута, в который назначена мельница.
    // -1 означает, что мельница пока не назначена.
    std::vector<int> obj_route;

    // obj_pos[obj] - позиция мельницы внутри маршрута.
    std::vector<int> obj_pos;

    // Число мельниц, уже назначенных в решение.
    int assigned_count = 0;

    static Solution empty(const DayData* data) {
        Solution s;
        s.data = data;
        s.routes.assign(data->num_heroes, {});
        s.route_costs.assign(data->num_heroes, 0);
        s.obj_route.assign(data->object_count, -1);
        s.obj_pos.assign(data->object_count, -1);
        return s;
    }

    bool assigned(int obj) const {
        return obj_route[obj] != -1;
    }

    int visited_count() const {
        return assigned_count;
    }

    // Tie-break:
    // если два решения собрали одинаковое число мельниц,
    // мы предпочитаем то, где осталось больше движения.
    int total_leftover() const {
        int sum = 0;
        for (int r = 0; r < data->num_heroes; ++r) {
            sum += std::max(0, data->hero_capacity(r) - route_costs[r]);
        }
        return sum;
    }

    // Сколько движения потрачено по всем героям.
    // Для статистики и анализа.
    int total_used() const {
        int sum = 0;
        for (int r = 0; r < data->num_heroes; ++r) {
            sum += std::min(data->hero_capacity(r), route_costs[r]);
        }
        return sum;
    }

    // --------------------------------------------------------
    // quality_key
    // --------------------------------------------------------
    //
    // Что делает:
    // Возвращает "ключ качества" решения, по которому мы сравниваем
    // два маршрута между собой.
    //
    // Качество определяется так:
    //   1. сначала максимизируем число посещённых мельниц;
    //   2. при равенстве максимизируем суммарный leftover.
    //
    // Почему именно так:
    // - главная цель - собрать как можно больше мельниц;
    // - leftover используется как tie-break:
    //   если две конфигурации собрали одинаковое число мельниц,
    //   предпочитаем ту, которая оставляет больше свободы по ходу.
    //
    // Это делает поиск чуть "аккуратнее":
    // алгоритм чаще предпочитает менее зажатые маршруты.
    std::pair<int, int> quality_key() const {
        return {visited_count(), total_leftover()};
    }

    // --------------------------------------------------------
    // update_index_from
    // --------------------------------------------------------
    //
    // Что делает:
    // После вставки или удаления мельницы обновляет быстрые индексы
    // obj_route и obj_pos для всех мельниц маршрута, начиная с позиции from.
    //
    // Чтобы такие операции были быстрыми, мы храним:
    //   obj_route[obj] - в каком маршруте находится мельница,
    //   obj_pos[obj]   - на какой позиции она стоит.
    //
    // Но после изменения маршрута позиции некоторых мельниц сдвигаются.
    // Поэтому индексы надо синхронизировать.
    //
    // Почему обновляем не весь маршрут, а только хвост начиная с from:
    // всё, что стоит до from, не изменило своей позиции,
    // а значит пересчитывать это не нужно.
    void update_index_from(int r, int from) {
        if (from < 0) from = 0;
        auto& route = routes[r];
        for (int pos = from; pos < (int)route.size(); ++pos) {
            int obj = route[pos];
            obj_route[obj] = r;
            obj_pos[obj] = pos;
        }
    }

    // --------------------------------------------------------
    // removal_delta_by_pos
    // --------------------------------------------------------
    //
    // Что делает:
    // Считает, на сколько уменьшится стоимость маршрута героя r,
    // если удалить мельницу из позиции pos.
    //
    // Что это даёт LNS:
    // Это симметричная идея к insertion_delta.
    // Вместо полного пересчёта маршрута мы смотрим только на локальное изменение.
    //
    // Зачем это нужно:
    // - destroy_worst использует эту величину, чтобы понять,
    //   какие мельницы "дорого" держать в маршруте;
    // - remove_object использует её, чтобы быстро обновить route_costs.
    //
    // Разбор случаев:
    //
    // 1. В маршруте одна мельница.
    //    Тогда после удаления маршрут становится пустым,
    //    и исчезает вся его стоимость.
    //
    // 2. Удаляем первую мельницу.
    //    Раньше маршрут начинался с x, а теперь начнётся с b.
    //    Значит:
    //      - убираем стартовую стоимость до x,
    //      + возвращаем стартовую стоимость до b,
    //      - убираем переход x -> b и VISIT_COST для x.
    //
    // 3. Удаляем последнюю мельницу.
    //    Просто убираем последний переход a -> x и VISIT_COST для x.
    //
    // 4. Удаляем мельницу из середины.
    //    Раньше был кусок a -> x -> b.
    //    После удаления он становится a -> b.
    //    Значит:
    //      - убираем dist(a, x),
    //      - убираем dist(x, b),
    //      + возвращаем dist(a, b),
    //      - убираем VISIT_COST для x.
    int removal_delta_by_pos(int r, int pos) const {
        const auto& route = routes[r];
        int n = (int)route.size();
        int x = route[pos];

        if (n == 1) {
            return route_costs[r];
        }

        if (pos == 0) {
            int b = route[1];
            return data->start_cost(r, x) - data->start_cost(r, b) + data->dist(x, b) + VISIT_COST;
        }

        if (pos == n - 1) {
            int a = route[n - 2];
            return data->dist(a, x) + VISIT_COST;
        }

        int a = route[pos - 1];
        int b = route[pos + 1];
        return data->dist(a, x) + data->dist(x, b) - data->dist(a, b) + VISIT_COST;
    }

    int removal_delta(int obj) const {
        return removal_delta_by_pos(obj_route[obj], obj_pos[obj]);
    }

    // --------------------------------------------------------
    // insertion_delta
    // --------------------------------------------------------
    //
    // Что делает:
    // Считает локальное изменение стоимости маршрута при вставке мельницы.
    //
    // "На сколько увеличится стоимость маршрута героя r,
    //  если вставить мельницу obj в позицию pos?"
    //
    //
    // Благодаря этому:
    // - greedy repair работает быстро;
    // - regret repair тоже остаётся приемлемым по времени.
    //
    // Разбираем случаи:
    //
    // 1. Маршрут пуст.
    //    Тогда стоимость вставки - это просто стартовая стоимость
    //    добраться до мельницы и посетить его.
    //
    // 2. Вставка в начало.
    //    Раньше первым была мельница b, теперь первым станет obj.
    //    Поэтому:
    //      + добавляем стоимость "старт -> obj"
    //      - убираем старую стоимость "старт -> b"
    //      + добавляем переход obj -> b и VISIT_COST для obj
    //
    // 3. Вставка в конец.
    //    Добавляем переход от последней мельницы к obj и VISIT_COST.
    //
    // 4. Вставка в середину.
    //    Раньше был кусок a -> b.
    //    После вставки он заменяется на a -> obj -> b.
    //    Поэтому:
    //      + dist(a, obj)
    //      + dist(obj, b)
    //      - dist(a, b)
    //      + VISIT_COST
    int insertion_delta(int r, int obj, int pos) const {
        const auto& route = routes[r];
        int n = (int)route.size();

        if (n == 0) {
            return data->start_cost(r, obj);
        }

        if (pos == 0) {
            int b = route[0];
            return data->start_cost(r, obj) - data->start_cost(r, b) + data->dist(obj, b) + VISIT_COST;
        }

        if (pos == n) {
            int a = route[n - 1];
            return data->dist(a, obj) + VISIT_COST;
        }

        int a = route[pos - 1];
        int b = route[pos];
        return data->dist(a, obj) + data->dist(obj, b) - data->dist(a, b) + VISIT_COST;
    }

    // --------------------------------------------------------
    // best_insertion_in_route
    // --------------------------------------------------------
    //
    // Эта функция ищет ЛУЧШЕЕ место вставки мельницы obj
    // в маршрут героя r.
    //
    // Что значит "лучшее":
    // - среди всех позиций 0..n выбираем ту,
    //   где insertion_delta минимальна.
    //
    // Но есть дополнительное ограничение:
    // - новый маршрут должен оставаться допустимым по ёмкости героя.
    //
    // Возвращаем:
    // - std::nullopt, если вставить мельницу в этот маршрут нельзя;
    // - пару (best_delta, best_pos), если можно.
    //
    // Что это даёт LNS:
    // Repair-фаза LNS должна уметь отвечать на вопрос:
    // "Куда выгоднее всего вставить эту мельницу?"
    //
    // Эта функция - базовый кирпич для repair-операторов.
    // И greedy, и regret2 опираются именно на неё.
    std::optional<std::pair<int, int>> best_insertion_in_route(int r, int obj) const {
        const auto& route = routes[r];
        int n = (int)route.size();

        int cap_ext = data->hero_capacity(r) + VISIT_COST;
        int base = route_costs[r];

        bool found = false;
        int best_delta = 0;
        int best_pos = -1;

        for (int pos = 0; pos <= n; ++pos) {
            int delta = insertion_delta(r, obj, pos);

            if (base + delta <= cap_ext) {
                if (!found || delta < best_delta || (delta == best_delta && pos < best_pos)) {
                    found = true;
                    best_delta = delta;
                    best_pos = pos;
                }
            }
        }

        if (!found) {
            return std::nullopt;
        }

        return std::make_pair(best_delta, best_pos);
    }

    // --------------------------------------------------------
    // insert
    // --------------------------------------------------------
    //
    // Реально вставляет мельницу в маршрут и обновляет все индексы.
    //
    // Что именно делает:
    // 1. Проверяет, что мельница ещё никому не назначена.
    // 2. Считает дельту стоимости (или берёт уже готовую, если она передана).
    // 3. Вставляет мельницу в вектор маршрута.
    // 4. Увеличивает кэшированную стоимость маршрута.
    // 5. Обновляет быстрые индексы obj_route / obj_pos.
    // 6. Увеличивает assigned_count.
    void insert(int obj, int r, int pos, std::optional<int> given_delta = std::nullopt) {
        if (assigned(obj)) {
            throw std::runtime_error("insert: мельница уже назначена");
        }

        int delta = given_delta.has_value() ? *given_delta : insertion_delta(r, obj, pos);

        routes[r].insert(routes[r].begin() + pos, obj);
        route_costs[r] += delta;
        obj_route[obj] = r;
        assigned_count++;

        update_index_from(r, pos);
    }

    // --------------------------------------------------------
    // remove_object
    // --------------------------------------------------------
    //
    // Реально удаляет мельницу из её маршрута.
    //
    // Что делает:
    // 1. По быстрым индексам находит маршрут и позицию мельницы.
    // 2. Считает, на сколько уменьшится стоимость маршрута.
    // 3. Удаляет мельницу из вектора маршрута.
    // 4. Уменьшает route_costs[r] на найденную дельту.
    // 5. Сбрасывает obj_route[obj] и obj_pos[obj].
    // 6. Обновляет индексы объектов, стоящих после удалённого.
    // 7. Уменьшает assigned_count.
    //
    // Возвращаемое значение:
    // - на сколько уменьшилась стоимость маршрута.
    //
    // Это и есть сердцевина large neighborhood search:
    // не мелкая перестановка, а удаление заметного куска решения
    // и последующая пересборка.
    int remove_object(int obj) {
        if (!assigned(obj)) return 0;

        int r = obj_route[obj];
        int pos = obj_pos[obj];
        int delta = removal_delta_by_pos(r, pos);

        routes[r].erase(routes[r].begin() + pos);
        route_costs[r] -= delta;
        obj_route[obj] = -1;
        obj_pos[obj] = -1;
        assigned_count--;

        update_index_from(r, pos);
        return delta;
    }

    // --------------------------------------------------------
    // validate_basic
    // --------------------------------------------------------
    //
    // Что проверяем:
    //
    // 1. route_costs[r] действительно совпадает с пересчётом route_cost(...).
    //    Это защищает нас от ошибок в дельтах вставки/удаления.
    //
    // 2. Стоимость маршрута не превышает hero_capacity + VISIT_COST.
    //    То есть решение удовлетворяет нашей упрощённой модели допустимости.
    //
    // 3. Каждая мельница:
    //    - имеет корректный индекс,
    //    - встречается не более одного раза,
    //    - согласован с таблицами obj_route и obj_pos.
    //
    // 4. assigned_count совпадает с реальным числом назначенных мельниц.
    bool validate_basic() const {
        std::vector<int> seen(data->object_count, 0);

        int cnt = 0;
        for (int r = 0; r < data->num_heroes; ++r) {
            int actual_cost = data->route_cost(r, routes[r]);

            if (actual_cost != route_costs[r]) return false;
            if (actual_cost > data->hero_capacity(r) + VISIT_COST) return false;

            for (int pos = 0; pos < (int)routes[r].size(); ++pos) {
                int obj = routes[r][pos];

                if (obj < 0 || obj >= data->object_count) return false;
                if (seen[obj]) return false;
                seen[obj] = 1;

                if (obj_route[obj] != r) return false;
                if (obj_pos[obj] != pos) return false;

                cnt++;
            }
        }

        return cnt == assigned_count;
    }
};

// ============================================================
// Операторы LNS
// ============================================================

// DestroyOp - оператор "разрушения" решения.
//
// На destroy-фазе мы намеренно убираем часть уже назначенных мельниц,
// чтобы затем repair-фаза попробовала собрать решение заново, но лучше.
enum class DestroyOp {
    // Случайное удаление:
    // хорошо добавляет разнообразие поиска.
    RANDOM = 0,

    // Удаление "плохих" мельниц:
    // направляет поиск туда, где можно сильнее перестроить решение.
    WORST = 1
};

// RepairOp - оператор "восстановления" решения.
//
// На repair-фазе мы пытаемся снова вставить удалённые и ещё не назначенные мельницы.
enum class RepairOp {
    // Самый дешёвый следующий шаг.
    GREEDY = 0,

    // Бережёт "хрупкие" мельницы, которые потом можно потерять.
    REGRET2 = 1
};

std::string destroy_op_to_string(DestroyOp op) {
    return op == DestroyOp::RANDOM ? "random" : "worst";
}

std::string repair_op_to_string(RepairOp op) {
    return op == RepairOp::GREEDY ? "greedy" : "regret2";
}

// ============================================================
// LNS-солвер одного дня
// ============================================================

class LNSSolver {
public:
    struct RunStats {
        int iterations_done = 0;
        int accepted_moves = 0;
        int improving_moves = 0;
        int best_updates = 0;
    };

    LNSSolver(const DayData& data, const Config& cfg, int seed, double day_time_limit)
        : m_data(data), m_cfg(cfg), m_rng(seed), m_day_time_limit(day_time_limit) {}

    // --------------------------------------------------------
    // solve
    // --------------------------------------------------------
    //
    // LNS = "разруши и почини" много раз подряд.
    //
    // current  - текущее решение, вокруг которого мы ищем.
    // best     - лучшее решение, найденное за всё время.
    //
    // На каждой итерации:
    // 1. создаём candidate = current;
    // 2. разрушаем candidate;
    // 3. чиним candidate;
    // 4. если candidate хороший - принимаем;
    // 5. если candidate лучший за всё время - запоминаем.
    //
    // Почему LNS вообще работает:
    //
    // - destroy ломает текущее решение и позволяет выйти из "ловушки";
    // - repair собирает его заново, потенциально в лучшей конфигурации;
    // - simulated annealing позволяет иногда принимать ухудшение,
    //   чтобы не застревать в локальном оптимуме слишком рано.
    std::pair<Solution, RunStats> solve(double deadline_sec) {
        if (m_data.object_count == 0) {
            return {Solution::empty(&m_data), m_stats};
        }

        // Строим стартовое решение.
        // Оно не обязано быть идеальным - достаточно, чтобы было корректным.
        Solution current = build_initial_solution();
        Solution best = current;

        double start = now_sec();

        while (now_sec() < deadline_sec &&
               (m_cfg.iterations <= 0 || m_stats.iterations_done < m_cfg.iterations)) {
            ++m_stats.iterations_done;

            // progress от 0 до 1 - как далеко мы продвинулись по времени.
            double elapsed = now_sec() - start;
            double progress = std::min(1.0, elapsed / std::max(1e-9, m_day_time_limit));

            // Температура падает со временем.
            double temperature = compute_temperature(progress);

            // Просто случайно выбираем destroy и repair.
            DestroyOp d_op = (rand01() < 0.5 ? DestroyOp::RANDOM : DestroyOp::WORST);
            RepairOp r_op = (rand01() < 0.5 ? RepairOp::GREEDY : RepairOp::REGRET2);

            // Работаем не с current напрямую, а с его копией.
            // Так проще сравнивать "до" и "после".
            Solution cand = current;

            int q = choose_q(cand);
            destroy(cand, d_op, q);
            repair(cand, r_op);

            auto cur_key = current.quality_key();
            auto cand_key = cand.quality_key();

            // Решаем, принять ли новый кандидат.
            // Даже если он хуже, при ненулевой температуре
            // можем иногда принять его - это и есть simulated annealing.
            if (accept(cand, current, temperature)) {
                if (better_key(cand_key, cur_key)) {
                    m_stats.improving_moves++;
                }

                current = std::move(cand);
                m_stats.accepted_moves++;

                // Отдельно запоминаем глобально лучшее решение.
                if (better_key(current.quality_key(), best.quality_key())) {
                    best = current;
                    m_stats.best_updates++;
                }
            }

            if (m_stats.iterations_done % m_cfg.log_every == 0) {
                log_msg(std::format(
                    "[day {}] iter={:<6} | destroy={:<7} | repair={:<7} | best=({:>3}, {:>6}) | cur=({:>3}, {:>6}) | temp={:.6f}",
                    m_data.day,
                    m_stats.iterations_done,
                    destroy_op_to_string(d_op),
                    repair_op_to_string(r_op),
                    best.visited_count(),
                    best.total_leftover(),
                    current.visited_count(),
                    current.total_leftover(),
                    temperature
                ));
            }
        }

        return {best, m_stats};
    }

private:
    const DayData& m_data;
    Config m_cfg;
    std::mt19937 m_rng;
    double m_day_time_limit;
    RunStats m_stats;

    bool better_key(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    }

    int randint(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(m_rng);
    }

    double rand01() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(m_rng);
    }

    double compute_temperature(double progress) const {
        if (m_cfg.temp_start <= 0.0 || m_cfg.temp_end <= 0.0) return 0.0;

        progress = std::clamp(progress, 0.0, 1.0);

        if (std::abs(m_cfg.temp_start - m_cfg.temp_end) < 1e-15) {
            return m_cfg.temp_start;
        }

        return m_cfg.temp_start * std::pow(m_cfg.temp_end / m_cfg.temp_start, progress);
    }

    // --------------------------------------------------------
    // choose_q
    // --------------------------------------------------------
    //
    // Что делает:
    // Выбирает, сколько мельниц удалить на текущей destroy-фазе.
    //
    // Что это даёт LNS:
    // В Large Neighborhood Search важен размер разрушения.
    //
    // Если удалять слишком мало:
    // - поиск будет слишком "локальным";
    // - алгоритм может застрять рядом с текущим решением.
    //
    // Если удалять слишком много:
    // - repair будет почти собирать решение заново;
    // - потеряется польза от уже найденной хорошей структуры.
    //
    // Поэтому мы удаляем не фиксированное число мельниц,
    // а случайное число q в диапазоне:
    //   [destroy_frac_min * visited_count,
    //    destroy_frac_max * visited_count]
    //
    // Это даёт баланс:
    // - иногда разрушение мягкое,
    // - иногда более сильное,
    // что делает поиск разнообразнее.
    int choose_q(const Solution& sol) {
        if (sol.visited_count() <= 0) return 0;

        int lo = std::max(1, (int)std::floor(m_cfg.destroy_frac_min * sol.visited_count()));
        int hi = std::max(lo, (int)std::ceil(m_cfg.destroy_frac_max * sol.visited_count()));
        hi = std::min(hi, sol.visited_count());
        lo = std::min(lo, hi);

        return randint(lo, hi);
    }

    // Если новое решение лучше - принимаем всегда.
    // Если хуже - иногда принимаем, особенно в начале поиска.
    bool accept(const Solution& cand, const Solution& cur, double temperature) {
        auto ck = cand.quality_key();
        auto uk = cur.quality_key();

        if (ck == uk || better_key(ck, uk)) {
            return true;
        }

        double delta =
            (cand.visited_count() - cur.visited_count()) +
            (double)(cand.total_leftover() - cur.total_leftover()) / 1000000.0;

        if (temperature <= 0.0) {
            return false;
        }

        double prob = std::exp(delta / temperature);
        return rand01() < prob;
    }

    Solution build_initial_solution() {
        // Начинаем с пустого решения
        // и жадно вставляем мельницы, пока можем.
        //
        // LNS не стартует "из ничего" - ему нужен хотя бы какой-то
        // корректный базовый маршрут, который потом можно перестраивать.
        Solution sol = Solution::empty(&m_data);
        repair_greedy(sol);

        log_msg(std::format(
            "[day {}] init  | visited={:<3} | leftover={:<6}",
            m_data.day,
            sol.visited_count(),
            sol.total_leftover()
        ));

        return sol;
    }

    void destroy(Solution& sol, DestroyOp op, int q) {
        if (q <= 0 || sol.visited_count() == 0) return;

        if (op == DestroyOp::RANDOM) {
            destroy_random(sol, q);
        } else {
            destroy_worst(sol, q);
        }
    }

    // Случайное разрушение.
    //
    // Что это даёт LNS:
    // Иногда полезно ломать решение без сильной логики,
    // просто чтобы попробовать совершенно другую конфигурацию.
    void destroy_random(Solution& sol, int q) {
        std::vector<int> assigned_objs;
        assigned_objs.reserve(sol.visited_count());

        for (int obj = 0; obj < m_data.object_count; ++obj) {
            if (sol.assigned(obj)) {
                assigned_objs.push_back(obj);
            }
        }

        std::shuffle(assigned_objs.begin(), assigned_objs.end(), m_rng);

        q = std::min(q, (int)assigned_objs.size());
        for (int i = 0; i < q; ++i) {
            sol.remove_object(assigned_objs[i]);
        }
    }

    // --------------------------------------------------------
    // destroy_worst
    // --------------------------------------------------------
    //
    // Что делает:
    // Удаляет мельницы, удаление которых сильнее всего "облегчает" маршруты.
    //
    // Что это даёт LNS:
    // Такой оператор разрушает решение не случайно, а осмысленно.
    // Он чаще выбивает мельницы, которые выглядят дорогими или неудобными,
    // а значит repair может потом собрать маршруты более удачно.
    //
    // Почему не выбираем всегда самую плохую мельницу:
    // чтобы сохранить разнообразие поиска.
    //
    // Поэтому мы:
    // 1. сортируем кандидатов по выгоде удаления;
    // 2. берём верхние rcl_size кандидатов;
    // 3. случайно выбираем одного из них.
    void destroy_worst(Solution& sol, int q) {
        for (int it = 0; it < q; ++it) {
            if (sol.visited_count() == 0) break;

            std::vector<std::pair<int, int>> cands;
            cands.reserve(sol.visited_count());

            for (int obj = 0; obj < m_data.object_count; ++obj) {
                if (!sol.assigned(obj)) continue;
                cands.push_back({sol.removal_delta(obj), obj});
            }

            std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });

            int limit = std::min((int)cands.size(), std::max(1, m_cfg.rcl_size));
            int idx = randint(0, limit - 1);
            sol.remove_object(cands[idx].second);
        }
    }

    void repair(Solution& sol, RepairOp op) {
        if (op == RepairOp::GREEDY) {
            repair_greedy(sol);
        } else {
            repair_regret2(sol);
        }
    }

    // --------------------------------------------------------
    // greedy_insert_one
    // --------------------------------------------------------
    //
    // Что делает:
    // Выбирает одну самую выгодную локальную вставку.
    //
    // Идея:
    // перебираем все ещё не назначенные мельницы и все маршруты,
    // для каждой мельницы ищем лучшую допустимую вставку,
    // а затем выбираем глобально самую дешёвую вставку.
    //
    // То есть на каждом шаге отвечаем на вопрос:
    // "Какую мельницу сейчас проще всего вставить в решение?"
    //
    // Это очень простой и понятный repair-оператор.
    //
    // Плюсы:
    // - легко понять;
    // - быстро даёт неплохие решения.
    //
    // Минусы:
    // - может "захватывать" лёгкие мельницы,
    //   оставляя сложные на потом, когда вставить их уже некуда.
    bool greedy_insert_one(Solution& sol) {
        int best_obj = -1;
        int best_r = -1;
        int best_pos = -1;
        int best_delta = 0;
        bool found = false;

        for (int obj = 0; obj < m_data.object_count; ++obj) {
            if (sol.assigned(obj)) continue;

            for (int r = 0; r < m_data.num_heroes; ++r) {
                auto ins = sol.best_insertion_in_route(r, obj);
                if (!ins.has_value()) continue;

                int delta = ins->first;
                int pos = ins->second;

                if (!found ||
                    delta < best_delta ||
                    (delta == best_delta && r < best_r) ||
                    (delta == best_delta && r == best_r && pos < best_pos) ||
                    (delta == best_delta && r == best_r && pos == best_pos && obj < best_obj)) {
                    found = true;
                    best_obj = obj;
                    best_r = r;
                    best_pos = pos;
                    best_delta = delta;
                }
            }
        }

        if (!found) {
            return false;
        }

        sol.insert(best_obj, best_r, best_pos, best_delta);
        return true;
    }

    void repair_greedy(Solution& sol) {
        while (greedy_insert_one(sol)) {}
    }

    // --------------------------------------------------------
    // regret2_insert_one
    // --------------------------------------------------------
    //
    // Что делает:
    // Выбирает не просто "самую дешёвую сейчас" вставку,
    // а мельницу, которую опасно откладывать на потом.
    //
    // Основная идея:
    // не все мельницы одинаково "срочные".
    //
    // Для одной мельницы может быть:
    // - один очень хороший вариант вставки,
    // - и второй вариант сильно хуже.
    //
    // Если такую мельницу не вставить сейчас,
    // позже хороший вариант может пропасть,
    // и мельницу станет трудно или невозможно вставить.
    //
    // Поэтому для каждой мельницы считаем:
    // - best1 = лучшая стоимость вставки;
    // - best2 = вторая лучшая стоимость вставки;
    // - regret = best2 - best1.
    //
    // Чем больше regret, тем важнее вставить мельницу прямо сейчас.
    //
    // На каждом шаге выбираем мельницу с максимальным regret.
    //
    // Плюсы по сравнению с greedy:
    // - чаще бережёт "хрупкие" мельницы,
    //   которые потом могут стать недоступны.
    //
    // Минусы:
    // - сложнее;
    // - обычно медленнее обычной жадной вставки.
    bool regret2_insert_one(Solution& sol) {
        int chosen_obj = -1;
        int chosen_r = -1;
        int chosen_pos = -1;
        int chosen_best_delta = 0;
        int best_regret = std::numeric_limits<int>::min();
        bool found = false;

        static constexpr int BIG_M = 1000000;

        for (int obj = 0; obj < m_data.object_count; ++obj) {
            if (sol.assigned(obj)) continue;

            int best1 = std::numeric_limits<int>::max();
            int best2 = std::numeric_limits<int>::max();
            int best_route = -1;
            int best_pos = -1;

            for (int r = 0; r < m_data.num_heroes; ++r) {
                auto ins = sol.best_insertion_in_route(r, obj);
                if (!ins.has_value()) continue;

                int delta = ins->first;
                int pos = ins->second;

                if (delta < best1) {
                    best2 = best1;
                    best1 = delta;
                    best_route = r;
                    best_pos = pos;
                } else if (delta < best2) {
                    best2 = delta;
                }
            }

            if (best_route == -1) continue;

            if (best2 == std::numeric_limits<int>::max()) {
                best2 = best1 + BIG_M;
            }

            int regret = best2 - best1;

            if (!found ||
                regret > best_regret ||
                (regret == best_regret && best1 < chosen_best_delta) ||
                (regret == best_regret && best1 == chosen_best_delta && obj < chosen_obj)) {
                found = true;
                best_regret = regret;
                chosen_obj = obj;
                chosen_r = best_route;
                chosen_pos = best_pos;
                chosen_best_delta = best1;
            }
        }

        if (!found) {
            return false;
        }

        sol.insert(chosen_obj, chosen_r, chosen_pos, chosen_best_delta);
        return true;
    }

    void repair_regret2(Solution& sol) {
        while (regret2_insert_one(sol)) {}
    }
};

// ============================================================
// Результат за неделю
// ============================================================

struct WeekResult {
    std::vector<std::vector<int>> submission_by_hero;
    int total_visited = 0;
    int total_leftover = 0;
    int total_used_moves = 0;
};

// ============================================================
// Сохранение результатов
// ============================================================

void save_submission_csv(
    const fs::path& path,
    int heroes_count,
    const std::vector<std::vector<int>>& submission_by_hero
) {
    std::ofstream fout(path);
    if (!fout) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + path.string());
    }

    fout << "hero_id,object_id\n";

    for (int h = 0; h < heroes_count; ++h) {
        int hero_id = h + 1;
        for (int object_id : submission_by_hero[h]) {
            fout << hero_id << "," << object_id << "\n";
        }
    }
}

void save_summary_txt(
    const fs::path& path,
    int total_visited,
    int total_used_moves,
    int total_leftover,
    int fixed_hero_cost
) {
    std::ofstream fout(path);
    if (!fout) {
        throw std::runtime_error("Не удалось открыть файл для записи: " + path.string());
    }

    int reward = total_visited * 500;
    int net_score = reward - fixed_hero_cost;

    fout << "visited_total=" << total_visited << "\n";
    fout << "used_moves_total=" << total_used_moves << "\n";
    fout << "leftover_total=" << total_leftover << "\n";
    fout << "reward=" << reward << "\n";
    fout << "fixed_hero_cost=" << fixed_hero_cost << "\n";
    fout << "net_score=" << net_score << "\n";
}

// ============================================================
// Решение всей недели
// ============================================================
//
// Между днями переносим:
// - где герой закончил день;
// - сколько движения условно "сохранилось".
WeekResult solve_week(const FullData& full, const Config& cfg) {
    if (cfg.heroes > full.hero_count()) {
        throw std::runtime_error("Запрошено героев больше, чем доступно");
    }

    WeekResult result;
    result.submission_by_hero.assign(cfg.heroes, {});

    std::vector<HeroState> hero_states(cfg.heroes);
    for (int h = 0; h < cfg.heroes; ++h) {
        hero_states[h].anchor_ext = 0;
        hero_states[h].carry_discount = 0;
    }

    for (int day = 1; day <= DAYS; ++day) {
        double day_limit = cfg.day_time_limits[day];

        log_msg(std::string(80, '-'));
        log_msg(std::format("DAY {} | time_limit={}", day, (int)day_limit));
        log_msg(std::string(80, '-'));

        DayData day_data = DayData::build_for_day(full, day, cfg.heroes, hero_states);

        Solution day_best = Solution::empty(&day_data);
        LNSSolver::RunStats day_stats{};

        if (day_limit > 0.0 && day_data.object_count > 0) {
            int run_seed = cfg.seed + day * 10000019;

            log_msg(std::format(
                "[day {}] seed={:<10} | mills={:<4}",
                day, run_seed, day_data.object_count
            ));

            LNSSolver solver(day_data, cfg, run_seed, day_limit);
            auto res = solver.solve(now_sec() + day_limit);
            day_best = std::move(res.first);
            day_stats = res.second;
        }

        if (!day_best.validate_basic()) {
            throw std::runtime_error("Решение дня " + std::to_string(day) + " не прошло базовую проверку");
        }

        // Переносим маршрут дня в общий недельный submission
        // и обновляем состояния героев на следующий день.
        for (int h = 0; h < day_data.num_heroes; ++h) {
            int cap = day_data.hero_capacity(h);

            for (int obj : day_best.routes[h]) {
                result.submission_by_hero[h].push_back(day_data.object_id(obj));
            }

            if (!day_best.routes[h].empty()) {
                int last_obj_internal = day_best.routes[h].back();
                hero_states[h].anchor_ext = day_data.object_id(last_obj_internal);
                hero_states[h].carry_discount = std::max(0, cap - day_best.route_costs[h]);
            } else {
                hero_states[h].carry_discount += cap;
            }
        }

        result.total_visited += day_best.visited_count();
        result.total_leftover += day_best.total_leftover();
        result.total_used_moves += day_best.total_used();

        log_msg(std::format(
            "[day {}] done   | visited={:<3} | leftover={:<6} | used_moves={:<6} | accepted={:<6} | improving={:<6} | best_updates={:<6}",
            day,
            day_best.visited_count(),
            day_best.total_leftover(),
            day_best.total_used(),
            day_stats.accepted_moves,
            day_stats.improving_moves,
            day_stats.best_updates
        ));
    }

    return result;
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    try {
        Config cfg;
        if (!parse_args(argc, argv, cfg)) {
            return 0;
        }

        fs::create_directories(cfg.output_dir);

        log_msg(std::string(80, '='));
        log_msg("LNS SOLVER");
        log_msg(std::string(80, '='));
        log_msg(std::format("Data dir: {}", cfg.data_dir.string()));
        log_msg(std::format("Output dir: {}", cfg.output_dir.string()));
        log_msg(std::format("Fixed heroes: {}", cfg.heroes));
        log_msg("Init: greedy");
        log_msg("Destroy ops: random, worst");
        log_msg("Repair ops: greedy, regret2");

        FullData full = FullData::load(cfg.data_dir);
        WeekResult result = solve_week(full, cfg);

        int fixed_hero_cost = cfg.heroes * HERO_COST;
        int reward = result.total_visited * 500;
        int net_score = reward - fixed_hero_cost;

        fs::path submission_path = cfg.output_dir / "submission.csv";
        fs::path summary_path = cfg.output_dir / "summary.txt";

        save_submission_csv(submission_path, cfg.heroes, result.submission_by_hero);
        save_summary_txt(summary_path,
                         result.total_visited,
                         result.total_used_moves,
                         result.total_leftover,
                         fixed_hero_cost);

        log_msg(std::format("CSV saved: {}", submission_path.string()));
        log_msg(std::format("Summary saved: {}", summary_path.string()));

        log_msg(std::string(80, '='));
        log_msg("ИТОГИ");
        log_msg(std::string(80, '='));
        log_msg(std::format("Visited total: {}", result.total_visited));
        log_msg(std::format("Total used moves: {}", result.total_used_moves));
        log_msg(std::format("Total leftover: {}", result.total_leftover));
        log_msg(std::format("Reward: {}", reward));
        log_msg(std::format("Fixed hero cost: {}", fixed_hero_cost));
        log_msg(std::format("Net score: {}", net_score));
        log_msg(std::string(80, '='));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << wall_timestamp() << " | ERROR | " << e.what() << "\n";
        return 1;
    }
}
