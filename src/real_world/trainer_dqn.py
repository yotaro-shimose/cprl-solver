import random

import dgl
import numpy as np

from src.problem.tsptw.learning.trainer_dqn import TrainerDQN
from src.problem.tsptw.learning.brain_dqn import BrainDQN
from src.real_world.environment import RealEnvironment
from src.real_world.generation import RealTSPTWGenerator
from src.real_world.property import RealDQNArgs
from src.util.prioritized_replay_memory import PrioritizedReplayMemory

#  definition of constants
MEMORY_CAPACITY = 50000
STEP_EPSILON = 5000.0
UPDATE_TARGET_FREQUENCY = 500
VALIDATION_SET_SIZE = 100


class RealTrainerDQN(TrainerDQN):
    def __init__(self, args: RealDQNArgs):
        """
        Initialization of the trainer
        :param args:  RealDQNArgs object taking hyperparameters and instance  configuration
        """

        self.args = args
        if self.args.seed != -1:
            np.random.seed(self.args.seed)
        self.instance_size = self.args.n_city
        # Because we begin at a given city, so we have 1 city less to visit
        self.n_action = self.instance_size - 1

        self.num_node_feats = 6
        self.num_edge_feats = 5

        self.reward_scaling = 0.001

        self.brain = BrainDQN(
            self.args, self.num_node_feats, self.num_edge_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)

        self.steps_done = 0
        self.init_memory_counter = 0

        if args.n_step == -1:  # We go until the end of the episode
            self.n_step = self.n_action
        else:
            self.n_step = self.args.n_step

        self.instance_generator = RealTSPTWGenerator(self.args)

        self.validation_set = self.instance_generator.generate_field_dataset(
            size=VALIDATION_SET_SIZE, seed=self.args.seed)
        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("[INFO] n_edge_feat: %d" % self.num_edge_feats)
        print("***********************************************************")

    def run_episode(self, episode_idx: int, memory_initialization: bool):
        """
        Run a single episode, either for initializing the memory (random episode in this case)
        or for training the model (following DQN algorithm)
        :param episode_idx: the index of the current episode done (without memory initialization)
        :param memory_initialization: True if it is for initializing the memory
        :return: the loss and the current beta of the softmax selection
        """

        #  Generate a random instance
        instance = self.instance_generator.generate_field_instance()

        env = RealEnvironment(
            instance,
            self.num_node_feats,
            self.num_edge_feats,
            self.reward_scaling,
            self.args.grid_size,
        )

        cur_state = env.get_initial_environment()

        graph_list = [dgl.DGLGraph()] * self.n_action
        rewards_vector = np.zeros(self.n_action)
        actions_vector = np.zeros(self.n_action, dtype=np.int16)
        available_vector = np.zeros((self.n_action, self.args.n_city))

        idx = 0
        total_loss = 0

        #  the current temperature for the softmax selection: increase from 0 to MAX_BETA
        temperature = max(0., min(self.args.max_softmax_beta,
                                  (episode_idx - 1) / STEP_EPSILON * self.args.max_softmax_beta))

        #  execute the episode
        while True:

            graph = env.make_nn_input(cur_state, self.args.mode)
            avail = env.get_valid_actions(cur_state)
            avail_idx = np.argwhere(avail == 1).reshape(-1)

            # if we are in the memory initialization phase, a random episode is selected
            if memory_initialization:
                action = random.choice(avail_idx)
            # otherwise, we do the softmax selection
            else:
                action = self.soft_select_action(graph, avail, temperature)

                # each time we do a step, we increase the counter
                self.steps_done += 1
                # we periodically synchronize the target network
                if self.steps_done % UPDATE_TARGET_FREQUENCY == 0:
                    self.brain.update_target_model()

            cur_state, reward = env.get_next_state_with_reward(
                cur_state, action)

            graph_list[idx] = graph
            rewards_vector[idx] = reward
            actions_vector[idx] = action
            available_vector[idx] = avail

            if cur_state.is_done():
                break

            idx += 1

        episode_last_idx = idx

        #  compute the n-step values
        for i in range(self.n_action):

            if i <= episode_last_idx:
                cur_graph = graph_list[i]
                cur_available = available_vector[i]
            else:
                cur_graph = graph_list[episode_last_idx]
                cur_available = available_vector[episode_last_idx]

            if i + self.n_step < self.n_action:
                next_graph = graph_list[i + self.n_step]
                next_available = available_vector[i + self.n_step]
            else:
                next_graph = dgl.DGLGraph()
                next_available = env.get_valid_actions(cur_state)

            #  a state correspond to the graph, with the nodes that we can still visit
            state_features = (cur_graph, cur_available)
            next_state_features = (next_graph, next_available)

            #  the n-step reward
            reward = sum(rewards_vector[i:i+self.n_step])
            action = actions_vector[i]

            sample = (state_features, action, reward, next_state_features)

            if memory_initialization:
                # the error of the replay memory is equals to the reward, at initialization
                error = abs(reward)
                self.init_memory_counter += 1
                step_loss = 0
            else:
                # feed the memory with the new samples
                x, y, errors = self.get_targets([(0, sample, 0)])
                error = errors[0]
                step_loss = self.learning()  # learning procedure

            self.memory.add(error, sample)

            total_loss += step_loss

        return total_loss, temperature

    def evaluate_instance(self, idx):
        """
        Evaluate an instance with the current model
        :param idx: the index of the instance in the validation set
        :return: the reward collected for this instance
        """

        instance = self.validation_set[idx]
        env = RealEnvironment(
            instance,
            self.num_node_feats,
            self.num_edge_feats,
            self.reward_scaling,
            self.args.grid_size,
        )
        cur_state = env.get_initial_environment()

        total_reward = 0

        while True:
            graph = env.make_nn_input(cur_state, self.args.mode)

            avail = env.get_valid_actions(cur_state)

            action = self.select_action(graph, avail)

            cur_state, reward = env.get_next_state_with_reward(
                cur_state, action)

            total_reward += reward

            if cur_state.is_done():
                break

        return total_reward
