from src.problem.tsptw.environment.environment import Environment
from src.problem.tsptw.environment.tsptw import TSPTW
import numpy as np


class RealEnvironment(Environment):
    def __init__(
        self,
        instance: TSPTW,
        n_node_feat: int,
        n_edge_feat: int,
        reward_scaling: float,
        grid_size: int,
    ):
        """
        Initialize the DP/RL environment
        :param instance: a TSPTW instance
        :param n_node_feat: number of features for the nodes
        :param n_edge_feat: number of features for the edges
        :param reward_scaling: value for scaling the reward
        :param grid_size: coordinates range (used for normalization purpose)
        """

        self.instance = instance
        self.n_node_feat = n_node_feat
        self.n_edge_feat = n_edge_feat
        self.reward_scaling = reward_scaling
        self.grid_size = grid_size

        self.max_dist = np.sqrt(self.grid_size ** 2 + self.grid_size ** 2)
        self.max_tw_value = self.instance.max_travel_time
        self.ub_cost = self.max_dist * self.instance.n_city

        self.edge_feat_tensor = self.instance.get_edge_feat_tensor(
            self.max_dist)
