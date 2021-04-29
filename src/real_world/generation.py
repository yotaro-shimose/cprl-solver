import random

import numpy as np

from src.real_world.conversion import to_virtual_time, to_real_coord, in_field, EIGHT_HOURS
from src.problem.tsptw.environment.tsptw import euclidian_distance, TSPTW


def generate_field_instance(
    n_city,
    grid_size,
    solvable=False,
    max_tw_gap=10,
    max_tw_size=10,
    is_integer_instance=True,
    max_travel_time=to_virtual_time(EIGHT_HOURS),
    tw_ratio=1.0,
    seed=-1
):
    rand = random.Random()
    if seed != -1:
        rand.seed(seed)

    # Create coordinates
    x_coord = list()
    y_coord = list()
    count = 0
    while count < n_city:
        x, y = rand.uniform(0, grid_size), rand.uniform(0, grid_size)
        real_x, real_y = to_real_coord(x, y, grid_size=grid_size)
        if in_field(real_x, real_y):
            x_coord.append(x)
            y_coord.append(y)
            count += 1

    travel_time = euclidian_distance(
        n_city, x_coord, y_coord, is_integer_instance)

    if solvable:
        max_travel_time = (n_city - 1) * (max_tw_gap + max_tw_size)
        time_windows = np.zeros((n_city, 2))
        random_solution = list(range(1, n_city))
        rand.shuffle(random_solution)

        random_solution = [0] + random_solution

        time_windows[0, :] = [0, max_travel_time]

        for i in range(1, n_city):

            prev_city = random_solution[i-1]
            cur_city = random_solution[i]

            cur_dist = travel_time[prev_city][cur_city]

            tw_lb_min = time_windows[prev_city, 0] + cur_dist

            rand_tw_lb = rand.uniform(tw_lb_min, tw_lb_min + max_tw_gap)
            rand_tw_ub = rand.uniform(rand_tw_lb, rand_tw_lb + max_tw_size)

            if is_integer_instance:
                rand_tw_lb = np.floor(rand_tw_lb)
                rand_tw_ub = np.ceil(rand_tw_ub)

            time_windows[cur_city, :] = [rand_tw_lb, rand_tw_ub]
        max_travel_time = (n_city - 1) * (max_tw_gap + max_tw_size)

    else:
        # Create time window options
        time_options = list()
        time = 0
        while time <= max_travel_time - max_tw_size:
            time_options.append([time, time + max_tw_size])
            time += max_tw_size

        # Assign time window
        n_time_window = n_city * tw_ratio + 1
        time_windows = list()
        time_windows.append([0, max_travel_time])
        for i in range(1, n_city):
            if i < n_time_window:
                time_window = time_options[np.random.randint(
                    len(time_options))]
            else:
                time_window = [0, max_travel_time]
            time_windows.append(time_window)
        time_windows = np.array(time_windows)

    return TSPTW(n_city, travel_time, x_coord, y_coord, time_windows, grid_size, max_travel_time)


def generate_field_dataset(size, *args, **kwargs):
    dataset = [generate_field_instance(*args, **kwargs) for _ in range(size)]
    return dataset
