import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_action(self, action_values):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(action_values))
        else:
            return np.argmax(action_values)
