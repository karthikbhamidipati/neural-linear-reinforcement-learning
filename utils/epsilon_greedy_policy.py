import numpy as np
from rl.policy import Policy

from utils.config import EPSILON_MAX, EPSILON_INTERVAL, EPSILON_GREEDY_FRAMES, EPSILON_MIN, SEED


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, epsilon_min, epsilon_interval, epsilon_greedy_frames, seed=None):
        super(EpsilonGreedyPolicy, self).__init__()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_interval / epsilon_greedy_frames
        self.random_state = np.random.RandomState(seed=seed)

    def select_action(self, q_values):
        assert q_values.ndim == 1

        if self.random_state.uniform(0, 1) < self.epsilon:
            action = self.random_state.randint(0, len(q_values))
        else:
            action = self.argmax_random(q_values)

        self.decay_epsilon()
        return action

    def argmax_random(self, actions):
        max_value = np.max(actions)
        max_indices = np.flatnonzero(max_value == actions)
        return self.random_state.choice(max_indices)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def get_config(self):
        config = super(EpsilonGreedyPolicy, self).get_config()
        config['epsilon'] = self.epsilon
        config['epsilon_min'] = self.epsilon_min
        config['epsilon_decay'] = self.epsilon_decay
        return config


def get_policy():
    return EpsilonGreedyPolicy(EPSILON_MAX, EPSILON_MIN, EPSILON_INTERVAL, EPSILON_GREEDY_FRAMES, SEED)
