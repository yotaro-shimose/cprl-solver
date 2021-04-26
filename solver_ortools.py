from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from src.real_world import (
    generate_field_instance,
    to_real_coord,
    to_virtual_coord,
    calc_lat_lon,
    REAL_X_RANGE,
    REAL_Y_RANGE,
    EIGHT_HOURS,
    to_real_time,
    to_virtual_time
)
from src.problem.tsptw.environment.tsptw import TSPTW
import numpy as np

MAX_WAIT = int(to_virtual_time(EIGHT_HOURS * 3))  # Infinite Waiting Time


def solve(instance: TSPTW, time_limit=6):
    # Create tsptw instance for or_tools
    data = dict()
    data["time_matrix"] = instance.travel_time
    data["num_vehicles"] = 1
    data["depot"] = 0
    data["time_windows"] = np.round(
        instance.time_windows).astype(np.int).tolist()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        MAX_WAIT,  # allow waiting time
        MAX_WAIT,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    def get_solution_list(manager, routing, solution):
        answer = list()
        if solution is not None:
            index = routing.Start(0)
            answer.append(index)
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                answer.append(manager.IndexToNode(index))
        return answer

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = time_limit

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    return get_solution_list(manager, routing, solution)


def solution_indicator(instance: TSPTW, solution: list):
    # 評価指標
    wait = 0
    delay = 0
    cur_time = 0

    pre_x, pre_y = instance.x_coord[solution[0]], instance.y_coord[solution[0]]
    for city in solution:
        x = instance.x_coord[city]
        y = instance.y_coord[city]
        window = instance.time_windows[city]  # [t_start, t_end]
        distance = ((x - pre_x) ** 2 + (y - pre_y) ** 2) ** 0.5
        tmp_time = cur_time + distance
        if tmp_time < window[0]:
            wait += window[0] - tmp_time
            cur_time = window[0]
        elif tmp_time > window[1]:
            delay += tmp_time - window[1]
            cur_time = tmp_time
        else:
            cur_time = tmp_time
        pre_x, pre_y = x, y
    return cur_time, wait, delay


if __name__ == '__main__':
    # Constant
    radius = 10
    n_city = 10
    grid_size = 100
    solvable = True
    time_window_size = 2 * 60  # 2時間

    instance = generate_field_instance(
        n_city, grid_size, solvable=solvable, max_tw_size=to_virtual_time(time_window_size))
    solution = solve(instance)
    if len(solution) > 2:
        time, wait, delay = solution_indicator(instance, solution)
        time, wait, delay = map(to_real_time, [time, wait, delay])
        time, wait, delay = map(round, [time, wait, delay])
        time, wait, delay = map(int, [time, wait, delay])

        print(f"配送時間：{time}分　待機時間：{wait}分　遅延時間：{delay}分")
    else:
        print("実行可能な配送経路を見つけられません")
