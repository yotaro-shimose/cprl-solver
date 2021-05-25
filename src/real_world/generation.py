import random
from typing import List

import numpy as np

from src.real_world.conversion import to_virtual_time, to_real_coord, in_field, EIGHT_HOURS
from src.real_world.property import RealDQNArgs
from src.problem.tsptw.environment.tsptw import euclidian_distance, TSPTW


class RealTSPTWGenerator:
    def __init__(
        self,
        args: RealDQNArgs,
    ):
        self.n_city = args.n_city
        self.grid_size = args.grid_size
        self.tw_size = args.tw_size
        self.is_integer_instance = True
        self.max_travel_time = args.max_travel_time
        self.tw_ratio = args.tw_ratio
        # Field[x][y] = True/False True if x, y is in field
        self.field = [[in_field(*to_real_coord(x, y, grid_size=args.grid_size))
                       for y in range(args.grid_size)]for x in range(args.grid_size)]

    def in_field(self, x: int, y: int) -> bool:
        return self.field[x][y]

    def generate_field_instance(self, seed: int = -1) -> TSPTW:
        rand = random.Random()
        if seed != -1:
            rand.seed(seed)
        # Create coordinates
        x_coord = list()
        y_coord = list()
        count = 0
        while count < self.n_city:
            x, y = rand.uniform(0, self.grid_size), rand.uniform(
                0, self.grid_size)
            if self.in_field(int(x), int(y)):
                x_coord.append(x)
                y_coord.append(y)
                count += 1

        travel_time = euclidian_distance(
            self.n_city, x_coord, y_coord, self.is_integer_instance)

        # Create time window options
        time_options = list()
        time = 0
        while time <= self.max_travel_time - self.tw_size:
            time_options.append([time, time + self.tw_size])
            time += self.tw_size

        # Assign time window
        n_time_window = self.n_city * self.tw_ratio + 1
        time_windows = list()
        time_windows.append([0, self.max_travel_time])
        for i in range(1, self.n_city):
            if i < n_time_window:
                time_window = time_options[np.random.randint(
                    len(time_options))]
            else:
                time_window = [0, self.max_travel_time]
            time_windows.append(time_window)
        time_windows = np.array(time_windows)

        return TSPTW(
            self.n_city,
            travel_time,
            x_coord,
            y_coord,
            time_windows,
            self.grid_size,
            self.max_travel_time
        )

    def generate_field_dataset(self, size: int, seed: int = -1) -> List[TSPTW]:
        rand = random.Random()
        if seed != -1:
            rand.seed(seed)
            seeds = [rand.randint(0, int(1e+8)) for _ in range(size)]
        else:
            seeds = [-1 for _ in range(size)]
        return [self.generate_field_instance(seed) for seed in seeds]


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
