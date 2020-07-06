"""
Agent that uses value iteration for learning. This includes
* Q-Learning
* SARSA
"""
import abc
from typing import cast
import numpy as np

from rlmarket.agent import Agent
from rlmarket.agent.q_function import QFunction
from rlmarket.environment import StateT


class ValueIterationAgent(Agent):
    """ ValueIterationAgent. Supports discrete state and action space """

    num_actions: int = None
    q_function: QFunction = None

    def __init__(self, eps_max: float = 1.0, eps_min: float = 0.05, warm_up_period: int = 500,
                 allow_exploration: bool = True, alpha: float = 0.1, gamma: float = 0.99) -> None:
        """
        We use annealing e-greedy.
        When allow_exploration is True,  SARSA. When it is false,
        """
        self.eps = eps_max
        self.eps_min = eps_min
        self.eps_drop = (eps_max - eps_min) / warm_up_period
        self.allow_exploration = allow_exploration
        self.alpha = alpha
        self.gamma = gamma

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        self.num_actions = num_actions
        self.initialize_q_function(state_dimension, num_actions)

    @abc.abstractmethod
    def initialize_q_function(self, state_dimension: int, num_actions: int) -> None:
        """ Instantiate self.q_function """

    def act(self, state: StateT) -> int:
        if np.random.random() < self.eps:
            return np.random.randint(self.num_actions)
        return cast(int, np.argmax(self.q_function[state]))

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        if self.allow_exploration and np.random.random() < self.eps:
            max_q = np.random.choice(self.q_function[new_state])
        else:
            max_q = np.max(self.q_function[new_state])

        self.q_function.update(state, action, reward + self.gamma * max_q, self.alpha)
        if self.eps > self.eps_min:
            self.eps -= self.eps_drop

    def disable_exploration(self):
        self.eps = 0
