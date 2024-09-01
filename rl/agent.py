import numpy as np
from rl.policies import EpsilonGreedyPolicy


class Agent:
    """
    Defines our base RL agent class.
    It's a simple agent designed to solve a problem from the class of multi-armed bandit problems.
    """
    def __init__(self, n_actions, initial_value, policy):
        """
        Initializes the agent.
        :param n_actions: The number of actions the agent can take.
        :param initial_value: The initial value for the action-values.
        :param policy: The policy to use.
        """
        self.n_actions = n_actions
        self.initial_value = initial_value
        self.policy = policy
        self.action_values = np.ones(n_actions) * initial_value
        self.action_counts = np.zeros(n_actions)
        self.action_total_rewards = np.zeros(n_actions)
    
    def get_action(self):
        """
        Selects the action to take.
        """
        return self.policy.get_action(self.action_values)
    
    def update(self, action, reward):
        """
        Updates the agent's knowledge based on the action taken and the reward received.
        """
        self.action_counts[action] += 1
        self.action_total_rewards[action] += reward
        self.action_values[action] = self.action_total_rewards[action] / self.action_counts[action]
    
    def reset(self):
        """
        Resets the agent's state.
        """
        self.action_values = np.ones(self.n_actions) * self.initial_value
        self.action_counts = np.zeros(self.n_actions)
