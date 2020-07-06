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
from rlmarket.market import StateT


class ValueIterationAgent(Agent):
    """ ValueIterationAgent. Supports discrete state and action space """

    num_actions: int = None
    q_function: QFunction = None

    def __init__(self, eps_curr: float = 0.1, eps_next: float = 0.1, alpha: float = 0.3, gamma: float = 0.99) -> None:
        """
        eps_now is used in the e-greedy policy for current state. eps_next is used for the next state.
        When eps_next = 0, we recover Q-learning algorithm.
        When eps_now = eps_next neq 0, we have SARSA algorithm
        """
        self.eps_curr = eps_curr
        self.eps_next = eps_next
        self.alpha = alpha
        self.gamma = gamma

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        self.num_actions = num_actions
        self.initialize_q_function(state_dimension, num_actions)

    @abc.abstractmethod
    def initialize_q_function(self, state_dimension: int, num_actions: int) -> None:
        """ Instantiate self.q_function """

    def act(self, state: StateT) -> int:
        if np.random.random() < self.eps_curr:
            return np.random.randint(self.num_actions)
        return cast(int, np.argmax(self.q_function[state]))

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        if np.random.random() < self.eps_next:
            max_q = np.random.choice(self.q_function[new_state])
        else:
            max_q = np.max(self.q_function[new_state])

        self.q_function.update(state, action, reward + self.gamma * max_q, self.alpha)

    def disable_exploration(self):
        self.eps_curr = 0
        self.eps_next = 0
