import os
import sys
from collections import defaultdict

import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

VISIT_COST = 100
HERO_COST = 2500
N_DAYS = 7
NUM_HEROES_DAY1 = 20
PENALTY_DAY1 = 50000
VEHICLE_FIXED_COST_DAYN = 100000
TIME_LIMIT_DAY1_SEC = 120
TIME_LIMIT_DAYN_SEC = 10
LAST_MOVE_BONUS = 100


def load_data(data_dir):
    heroes = pd.read_csv(os.path.join(data_dir, "data_heroes.csv"))
    objects = pd.read_csv(os.path.join(data_dir, "data_objects.csv"))
    dist_start = pd.read_csv(os.path.join(data_dir, "dist_start.csv"))
    dist_objects = pd.read_csv(os.path.join(data_dir, "dist_objects.csv"))
    col_names = [c for c in dist_objects.columns if c.startswith("object_")]
    dist_mat = dist_objects[col_names].values.astype(np.int64)
    dist_start_dict = dict(zip(dist_start["object_id"], dist_start["dist_start"]))
    return {"heroes": heroes, "objects": objects, "dist_start": dist_start_dict, "dist_matrix": dist_mat}


def dist_between(data, from_id, to_id):
    if from_id == 0:
        return data["dist_start"].get(to_id, 0)
    if to_id == 0:
        return data["dist_start"].get(from_id, 0)
    return data["dist_matrix"][from_id - 1, to_id - 1]


def simulate_solution(data, solution):
    if not solution:
        return -0 * HERO_COST, {"reward": 0, "hero_cost": 0, "max_hero_id": 0}
    heroes_df = data["heroes"].set_index("hero_id")
    objects_df = data["objects"].set_index("object_id")
    routes = defaultdict(list)
    for hid, oid in solution:
        routes[hid].append(oid)
    max_hero_id = max(routes.keys())
    total_reward = 0
    for hid in sorted(routes.keys()):
        move_points = int(heroes_df.loc[hid, "move_points"])
        route = routes[hid]
        day, pos, mp_left = 1, 0, move_points
        for oid in route:
            day_open = int(objects_df.loc[oid, "day_open"])
            reward = int(objects_df.loc[oid, "reward"])
            travel = dist_between(data, pos, oid)
            while day <= N_DAYS:
                if travel <= mp_left:
                    break
                travel -= mp_left
                day += 1
                mp_left = move_points
            if day > N_DAYS:
                break
            if day < day_open:
                day, mp_left = day_open, move_points
            if mp_left >= VISIT_COST:
                mp_left -= VISIT_COST
            else:
                mp_left = 0
            if day == day_open:
                total_reward += reward
            pos = oid
    score = total_reward - max_hero_id * HERO_COST
    return score, {"reward": total_reward, "hero_cost": max_hero_id * HERO_COST, "max_hero_id": max_hero_id}


def solution_to_csv(solution, out_path):
    pd.DataFrame(solution, columns=["hero_id", "object_id"]).to_csv(out_path, index=False, encoding="utf-8")


def build_full_distance_matrix(data):
    dist_mat, dist_start, n = data["dist_matrix"], data["dist_start"], 700
    matrix = np.zeros((n + 1, n + 1), dtype=np.int64)
    matrix[0, 0] = 0
    for j in range(1, n + 1):
        matrix[0, j] = matrix[j, 0] = dist_start.get(j, 0)
    matrix[1:, 1:] = dist_mat
    return matrix


def add_visit_cost_to_matrix(matrix, depot=0):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if j == depot:
                matrix[i][j] = 0
            else:
                matrix[i][j] = int(matrix[i][j]) + VISIT_COST


def get_day_object_ids(objects_df, day):
    return sorted(objects_df[objects_df["day_open"] == day]["object_id"].tolist())


def submatrix_for_objects(full_matrix, object_ids):
    idx = [0] + object_ids
    return full_matrix[np.ix_(idx, idx)].copy()


def solve_vrp_day1(data, full_matrix, objects_df, num_heroes=None, time_limit_sec=None, penalty_day1=None):
    open_idx = get_day_object_ids(objects_df, 1)
    if not open_idx:
        return [], {}
    matrix = submatrix_for_objects(full_matrix, open_idx)
    add_visit_cost_to_matrix(matrix, depot=0)
    matrix = matrix.tolist()
    N = num_heroes if num_heroes is not None else NUM_HEROES_DAY1
    max_distances = (
        data["heroes"].sort_values("hero_id").head(N)["move_points"].add(LAST_MOVE_BONUS).tolist()
    )
    manager = pywrapcp.RoutingIndexManager(len(matrix), N, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    penalty = penalty_day1 if penalty_day1 is not None else PENALTY_DAY1
    for node in range(1, len(matrix)):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    routing.AddDimensionWithVehicleCapacity(transit_callback_index, 0, max_distances, True, "Distance")
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit_sec if time_limit_sec is not None else TIME_LIMIT_DAY1_SEC
    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return [], {}
    routes = []
    for vehicle_id in range(N):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    routes = [r for r in routes if len(r) > 1 and (set(r) - {0})]
    hero_end_positions = {}
    for hero_idx, route in enumerate(routes):
        last_node = 0
        for node in reversed(route):
            if node != 0:
                last_node = node
                break
        if last_node > 0 and last_node <= len(open_idx):
            hero_end_positions[hero_idx] = open_idx[last_node - 1]
        else:
            hero_end_positions[hero_idx] = 0
    return routes, hero_end_positions


def create_vehicle_matrices_for_day(data, objects_df, day, hero_end_positions, previous_routes):
    dist_mat = data["dist_matrix"]
    dist_start = data["dist_start"]
    open_idx = get_day_object_ids(objects_df, day)
    if not open_idx:
        return []
    previous_empty = {}
    if previous_routes:
        for hero_idx, route in enumerate(previous_routes):
            previous_empty[hero_idx] = len(route) <= 1
    vehicle_matrices = []
    for hero_idx in range(len(hero_end_positions)):
        end_pos = hero_end_positions.get(hero_idx, 0)
        stood_still = previous_empty.get(hero_idx, False)
        if stood_still:
            depot_dist = np.zeros(700, dtype=np.int64)
        elif end_pos == 0:
            depot_dist = np.array([dist_start.get(i, 0) for i in range(1, 701)], dtype=np.int64)
        else:
            depot_dist = dist_mat[end_pos - 1, :].copy()
        n = 700
        mat = np.zeros((n + 1, n + 1), dtype=np.int64)
        mat[0, 0] = 0
        for j in range(1, n + 1):
            mat[0, j] = depot_dist[j - 1]
            mat[j, 0] = depot_dist[j - 1]
        mat[1:, 1:] = dist_mat
        idx = [0] + open_idx
        sub = mat[np.ix_(idx, idx)].copy()
        add_visit_cost_to_matrix(sub, depot=0)
        vehicle_matrices.append(sub.tolist())
    return vehicle_matrices


def solve_vrp_day_n(data, objects_df, day, hero_end_positions, previous_routes, time_limit_sec=None):
    num_heroes = len(hero_end_positions)
    vehicle_matrices = create_vehicle_matrices_for_day(
        data, objects_df, day, hero_end_positions, previous_routes
    )
    if not vehicle_matrices:
        return [[] for _ in range(num_heroes)], dict(hero_end_positions)
    matrix_size = len(vehicle_matrices[0])
    manager = pywrapcp.RoutingIndexManager(matrix_size, num_heroes, 0)
    routing = pywrapcp.RoutingModel(manager)

    def make_callback(vehicle_id):
        def callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return vehicle_matrices[vehicle_id][from_node][to_node]
        return callback

    transit_indices = []
    for vehicle_id in range(num_heroes):
        cb = make_callback(vehicle_id)
        idx = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfVehicle(idx, vehicle_id)
        transit_indices.append(idx)
    for vehicle_id in range(num_heroes):
        routing.SetFixedCostOfVehicle(VEHICLE_FIXED_COST_DAYN, vehicle_id)
    max_distances = (
        data["heroes"]
        .sort_values("hero_id")
        .head(num_heroes)["move_points"]
        .add(LAST_MOVE_BONUS)
        .tolist()
    )
    routing.AddDimensionWithVehicleTransitAndCapacity(
        transit_indices, 0, max_distances, True, "Distance"
    )
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit_sec if time_limit_sec is not None else TIME_LIMIT_DAYN_SEC
    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return [[] for _ in range(num_heroes)], dict(hero_end_positions)
    routes = []
    for vehicle_id in range(num_heroes):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    open_idx = get_day_object_ids(objects_df, day)
    new_hero_end_positions = {}
    for vehicle_id in range(num_heroes):
        route = routes[vehicle_id]
        if len(route) > 1:
            last_node = 0
            for node in reversed(route):
                if node != 0:
                    last_node = node
                    break
            if last_node > 0 and last_node <= len(open_idx):
                new_hero_end_positions[vehicle_id] = open_idx[last_node - 1]
            else:
                new_hero_end_positions[vehicle_id] = hero_end_positions.get(vehicle_id, 0)
        else:
            new_hero_end_positions[vehicle_id] = hero_end_positions.get(vehicle_id, 0)
    return routes, new_hero_end_positions


def route_to_object_ids(route, day, objects_df):
    if not route or len(route) <= 1:
        return []
    open_idx = get_day_object_ids(objects_df, day)
    return [open_idx[node - 1] for node in route if node != 0 and 1 <= node <= len(open_idx)]


def pipeline_to_solution(data, all_routes_by_day, objects_df):
    solution = []
    for day_one_index, day_routes in enumerate(all_routes_by_day, start=1):
        for hero_idx, route in enumerate(day_routes):
            object_ids = route_to_object_ids(route, day_one_index, objects_df)
            hero_id = hero_idx + 1
            for oid in object_ids:
                solution.append((hero_id, oid))
    return solution


def run_pipeline(data_dir, verbose=True, num_heroes_day1=None, time_limit_day1_sec=None, time_limit_dayn_sec=None, penalty_day1=None):
    data = load_data(data_dir)
    objects_df = data["objects"]
    full_matrix = build_full_distance_matrix(data)
    n_h = num_heroes_day1 if num_heroes_day1 is not None else NUM_HEROES_DAY1
    if verbose:
        print(f"Day 1: {n_h} heroes")
    routes_day1, hero_end_positions = solve_vrp_day1(
        data, full_matrix, objects_df,
        num_heroes=num_heroes_day1,
        time_limit_sec=time_limit_day1_sec,
        penalty_day1=penalty_day1,
    )
    if not routes_day1:
        if verbose:
            print("Day 1: no solution")
        return []
    if verbose:
        print(f"  Active: {len(routes_day1)}")
    all_routes = [routes_day1]
    for day in range(2, N_DAYS + 1):
        if verbose:
            print(f"Day {day}")
        prev_routes = all_routes[-1]
        routes_d, hero_end_positions = solve_vrp_day_n(
            data, objects_df, day, hero_end_positions, prev_routes,
            time_limit_sec=time_limit_dayn_sec,
        )
        all_routes.append(routes_d)
    return pipeline_to_solution(data, all_routes, objects_df)


def solution_to_routes(solution):
    routes = defaultdict(list)
    for hid, oid in solution:
        routes[hid].append(oid)
    return dict(routes)


def routes_to_solution(routes):
    return [(h, oid) for h in sorted(routes) for oid in routes[h]]


def try_merge_last_hero(data, solution):
    routes = solution_to_routes(solution)
    max_h = max(routes.keys()) if routes else 0
    if max_h <= 1:
        return solution
    moved = routes.pop(max_h, [])
    routes[max_h - 1] = routes.get(max_h - 1, []) + moved
    new_sol = routes_to_solution(routes)
    score_old, _ = simulate_solution(data, solution)
    score_new, _ = simulate_solution(data, new_sol)
    return new_sol if score_new > score_old else solution


def merge_until_stable(data, solution):
    while True:
        new_sol = try_merge_last_hero(data, solution)
        if new_sol == solution:
            return solution
        solution = new_sol


CONFIGS_FULL = [
    (20, 120, 10), (20, 180, 15), (19, 120, 10), (19, 180, 15),
    (21, 120, 10), (21, 180, 15), (18, 180, 15),
]
CONFIGS_FAST = [(20, 180, 15), (19, 180, 15), (21, 120, 10)]


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(data_dir, "submission.csv")
    fast = "--fast" in sys.argv
    configs = CONFIGS_FAST if fast else CONFIGS_FULL
    print(f"Configs: {len(configs)}", flush=True)
    data = load_data(data_dir)
    best_solution = None
    best_score = -(10 ** 9)
    for i, (num_heroes, tl1, tln) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] H={num_heroes} {tl1}s/{tln}s ... ", end="", flush=True)
        try:
            solution = run_pipeline(
                data_dir, verbose=False,
                num_heroes_day1=num_heroes,
                time_limit_day1_sec=tl1,
                time_limit_dayn_sec=tln,
            )
            if not solution:
                print("no solution")
                continue
            solution = merge_until_stable(data, solution)
            score, info = simulate_solution(data, solution)
            n_visits = len(set(o for _, o in solution))
            marker = " *" if score > best_score else ""
            if score > best_score:
                best_score, best_solution = score, solution
                solution_to_csv(solution, out_path)
            print(f"score={score} reward={info['reward']} max_h={info['max_hero_id']} visits={n_visits}{marker}")
        except Exception as e:
            print(f"error: {e}")
    if best_solution is None:
        solution = run_pipeline(data_dir, verbose=True)
        if solution:
            best_solution = merge_until_stable(data, solution)
            best_score, _ = simulate_solution(data, best_solution)
            solution_to_csv(best_solution, out_path)
    if best_solution is not None:
        score, info = simulate_solution(data, best_solution)
        print("")
        print(f"Score: {score}")
        print(f"Reward: {info['reward']}  Max hero: {info['max_hero_id']}  Visits: {len(set(o for _, o in best_solution))}")
        print(out_path)
    else:
        print("No solution.")


if __name__ == "__main__":
    main()
