import pandas as pd
import numpy as np
import random
from src.problem.tsptw.environment.tsptw import euclidian_distance, TSPTW

EIGHT_HOURS = 8 * 60

RADIUS = 250
PATH = 'data/csv/longi_lati.csv'
VEHICLE_SPEED = 8 / 60 * 1000  # 時速8km/hを分速換算(現実世界における速度)
# Open Field Plot Data
with open(PATH, 'r') as file:
    df = pd.read_csv(file)
field_x, field_y = df["Ｘ座標"], df["Ｙ座標"]
REAL_X_RANGE = np.array([np.min(field_x), np.max(field_x)])
REAL_Y_RANGE = np.array([np.min(field_y), np.max(field_y)])


def in_field(x, y):
    for x_f, y_f in zip(field_x, field_y):
        if (x_f - x) ** 2 + (y_f - y) ** 2 < RADIUS ** 2:
            return True
    return False


def calc_xy(phi_deg, lambda_deg, phi0_deg=36, lambda0_deg=139+50./60):
    """ 緯度経度を平面直角座標に変換する
    - input:
        (phi_deg, lambda_deg): 変換したい緯度・経度[度]（分・秒でなく小数であることに注意）
        (phi0_deg, lambda0_deg): 平面直角座標系原点の緯度・経度[度]（分・秒でなく小数であることに注意）
    - output:
        x: 変換後の平面直角座標[m]
        y: 変換後の平面直角座標[m]
    """
    # 緯度経度・平面直角座標系原点をラジアンに直す
    phi_rad = np.deg2rad(phi_deg)
    lambda_rad = np.deg2rad(lambda_deg)
    phi0_rad = np.deg2rad(phi0_deg)
    lambda0_rad = np.deg2rad(lambda0_deg)

    # 補助関数
    def A_array(n):
        A0 = 1 + (n**2)/4. + (n**4)/64.
        A1 = -     (3./2)*(n - (n**3)/8. - (n**5)/64.)
        A2 = (15./16)*(n**2 - (n**4)/4.)
        A3 = -   (35./48)*(n**3 - (5./16)*(n**5))
        A4 = (315./512)*(n**4)
        A5 = -(693./1280)*(n**5)
        return np.array([A0, A1, A2, A3, A4, A5])

    def alpha_array(n):
        a0 = np.nan  # dummy
        a1 = (1./2)*n - (2./3)*(n**2) + (5./16)*(n**3) + \
            (41./180)*(n**4) - (127./288)*(n**5)
        a2 = (13./48)*(n**2) - (3./5)*(n**3) + \
            (557./1440)*(n**4) + (281./630)*(n**5)
        a3 = (61./240)*(n**3) - (103./140)*(n**4) + (15061./26880)*(n**5)
        a4 = (49561./161280)*(n**4) - (179./168)*(n**5)
        a5 = (34729./80640)*(n**5)
        return np.array([a0, a1, a2, a3, a4, a5])

    # 定数 (a, F: 世界測地系-測地基準系1980（GRS80）楕円体)
    m0 = 0.9999
    a = 6378137.
    F = 298.257222101

    # (1) n, A_i, alpha_iの計算
    n = 1. / (2*F - 1)
    A_array = A_array(n)
    alpha_array = alpha_array(n)

    # (2), S, Aの計算
    A_ = ((m0*a)/(1.+n))*A_array[0]  # [m]
    S_ = ((m0*a)/(1.+n))*(A_array[0]*phi0_rad +
                          np.dot(A_array[1:], np.sin(2*phi0_rad*np.arange(1, 6))))  # [m]

    # (3) lambda_c, lambda_sの計算
    lambda_c = np.cos(lambda_rad - lambda0_rad)
    lambda_s = np.sin(lambda_rad - lambda0_rad)

    # (4) t, t_の計算
    t = np.sinh(np.arctanh(np.sin(phi_rad)) - ((2*np.sqrt(n)) / (1+n))
                * np.arctanh(((2*np.sqrt(n)) / (1+n)) * np.sin(phi_rad)))
    t_ = np.sqrt(1 + t*t)

    # (5) xi', eta'の計算
    xi2 = np.arctan(t / lambda_c)  # [rad]
    eta2 = np.arctanh(lambda_s / t_)

    # (6) x, yの計算
    x = A_ * (xi2 + np.sum(np.multiply(alpha_array[1:],
                                       np.multiply(np.sin(2*xi2*np.arange(1, 6)),
                                                   np.cosh(2*eta2*np.arange(1, 6)))))) - S_  # [m]
    y = A_ * (eta2 + np.sum(np.multiply(alpha_array[1:],
                                        np.multiply(np.cos(2*xi2*np.arange(1, 6)),
                                                    np.sinh(2*eta2*np.arange(1, 6))))))  # [m]
    # return
    return x, y  # [m]


def calc_lat_lon(x, y, phi0_deg=36, lambda0_deg=139+50./60):
    """ 平面直角座標を緯度経度に変換する
    - input:
        (x, y): 変換したいx, y座標[m]
        (phi0_deg, lambda0_deg): 平面直角座標系原点の緯度・経度[度]（分・秒でなく小数であることに注意）
    - output:
        latitude:  緯度[度]
        longitude: 経度[度]
        * 小数点以下は分・秒ではないことに注意
    """
    # 平面直角座標系原点をラジアンに直す
    phi0_rad = np.deg2rad(phi0_deg)
    lambda0_rad = np.deg2rad(lambda0_deg)

    # 補助関数
    def A_array(n):
        A0 = 1 + (n**2)/4. + (n**4)/64.
        A1 = -     (3./2)*(n - (n**3)/8. - (n**5)/64.)
        A2 = (15./16)*(n**2 - (n**4)/4.)
        A3 = -   (35./48)*(n**3 - (5./16)*(n**5))
        A4 = (315./512)*(n**4)
        A5 = -(693./1280)*(n**5)
        return np.array([A0, A1, A2, A3, A4, A5])

    def beta_array(n):
        b0 = np.nan  # dummy
        b1 = (1./2)*n - (2./3)*(n**2) + (37./96) * \
            (n**3) - (1./360)*(n**4) - (81./512)*(n**5)
        b2 = (1./48)*(n**2) + (1./15)*(n**3) - \
            (437./1440)*(n**4) + (46./105)*(n**5)
        b3 = (17./480)*(n**3) - (37./840)*(n**4) - (209./4480)*(n**5)
        b4 = (4397./161280)*(n**4) - (11./504)*(n**5)
        b5 = (4583./161280)*(n**5)
        return np.array([b0, b1, b2, b3, b4, b5])

    def delta_array(n):
        d0 = np.nan  # dummy
        d1 = 2.*n - (2./3)*(n**2) - 2.*(n**3) + (116./45) * \
            (n**4) + (26./45)*(n**5) - (2854./675)*(n**6)
        d2 = (7./3)*(n**2) - (8./5)*(n**3) - (227./45) * \
            (n**4) + (2704./315)*(n**5) + (2323./945)*(n**6)
        d3 = (56./15)*(n**3) - (136./35)*(n**4) - \
            (1262./105)*(n**5) + (73814./2835)*(n**6)
        d4 = (4279./630)*(n**4) - (332./35)*(n**5) - (399572./14175)*(n**6)
        d5 = (4174./315)*(n**5) - (144838./6237)*(n**6)
        d6 = (601676./22275)*(n**6)
        return np.array([d0, d1, d2, d3, d4, d5, d6])

    # 定数 (a, F: 世界測地系-測地基準系1980（GRS80）楕円体)
    m0 = 0.9999
    a = 6378137.
    F = 298.257222101

    # (1) n, A_i, beta_i, delta_iの計算
    n = 1. / (2*F - 1)
    A_array = A_array(n)
    beta_array = beta_array(n)
    delta_array = delta_array(n)

    # (2), S, Aの計算
    A_ = ((m0*a)/(1.+n))*A_array[0]
    S_ = ((m0*a)/(1.+n))*(A_array[0]*phi0_rad +
                          np.dot(A_array[1:], np.sin(2*phi0_rad*np.arange(1, 6))))

    # (3) xi, etaの計算
    xi = (x + S_) / A_
    eta = y / A_

    # (4) xi', eta'の計算
    xi2 = xi - np.sum(np.multiply(beta_array[1:],
                                  np.multiply(np.sin(2*xi*np.arange(1, 6)),
                                              np.cosh(2*eta*np.arange(1, 6)))))
    eta2 = eta - np.sum(np.multiply(beta_array[1:],
                                    np.multiply(np.cos(2*xi*np.arange(1, 6)),
                                                np.sinh(2*eta*np.arange(1, 6)))))

    # (5) chiの計算
    chi = np.arcsin(np.sin(xi2)/np.cosh(eta2))  # [rad]
    latitude = chi + np.dot(delta_array[1:],
                            np.sin(2*chi*np.arange(1, 7)))  # [rad]

    # (6) 緯度(latitude), 経度(longitude)の計算
    longitude = lambda0_rad + np.arctan(np.sinh(eta2)/np.cos(xi2))  # [rad]

    # ラジアンを度になおしてreturn
    return np.rad2deg(latitude), np.rad2deg(longitude)  # [deg]


def to_real_coord(virtual_x, virtual_y, grid_size=100):
    x_range = REAL_X_RANGE[1] - REAL_X_RANGE[0]
    y_range = REAL_Y_RANGE[1] - REAL_Y_RANGE[0]
    map_range = max(x_range, y_range)
    real_x = virtual_x * map_range / grid_size + REAL_X_RANGE[0]
    real_y = virtual_y * map_range / grid_size + REAL_Y_RANGE[0]
    return real_x, real_y


def to_virtual_coord(real_x, real_y, grid_size=100):
    x_range = REAL_X_RANGE[1] - REAL_X_RANGE[0]
    y_range = REAL_Y_RANGE[1] - REAL_Y_RANGE[0]
    map_range = max(x_range, y_range)
    virtual_x = real_x / map_range * grid_size + REAL_X_RANGE[0]
    virtual_y = real_y / map_range * grid_size + REAL_Y_RANGE[0]
    return virtual_x, virtual_y


def to_real_time(virtual_time, grid_size=100):
    x_range = REAL_X_RANGE[1] - REAL_X_RANGE[0]
    y_range = REAL_Y_RANGE[1] - REAL_Y_RANGE[0]
    map_range = max(x_range, y_range)
    return virtual_time * map_range / grid_size / VEHICLE_SPEED


def to_virtual_time(real_time, grid_size=100):
    x_range = REAL_X_RANGE[1] - REAL_X_RANGE[0]
    y_range = REAL_Y_RANGE[1] - REAL_Y_RANGE[0]
    map_range = max(x_range, y_range)
    return real_time / map_range * grid_size * VEHICLE_SPEED


def generate_field_instance(n_city, grid_size, solvable=False, max_tw_gap=10, max_tw_size=10,
                            is_integer_instance=True, seed=-1):
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
        time_windows = np.zeros((n_city, 2))
        random_solution = list(range(1, n_city))
        rand.shuffle(random_solution)

        random_solution = [0] + random_solution

        time_windows[0, :] = [0, to_virtual_time(EIGHT_HOURS)]

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

    else:
        # Create time window options
        time_options = list()
        time = 0
        while time <= to_virtual_time(EIGHT_HOURS) - max_tw_size:
            time_options.append([time, time + max_tw_size])
            time += max_tw_size
        time_windows = list()
        time_windows.append([0, to_virtual_time(EIGHT_HOURS)])
        for i in range(1, n_city):
            time_window = time_options[np.random.randint(len(time_options))]
            time_windows.append(time_window)
        time_windows = np.array(time_windows)

    return TSPTW(n_city, travel_time, x_coord, y_coord, time_windows)
