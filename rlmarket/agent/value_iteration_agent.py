"""
Agent that uses value iteration for learning. This includes
* Q-Learning
* SARSA
"""
from typing import DefaultDict, cast
from collections import defaultdict
from functools import partial
import numpy as np

from rlmarket.agent import Agent
from rlmarket.market import StateT


class ValueIterationAgent(Agent):
    """ ValueIterationAgent. Supports discrete state and action space """

    num_actions: int = None
    q_function: DefaultDict[StateT, np.ndarray] = None

    def __init__(self, eps_now: float = 0.1, eps_next: float = 0.1, alpha: float = 0.3, gamma: float = 0.99) -> None:
        """
        eps_now is used in the e-greedy policy for current state. eps_next is used for the next state.
        When eps_next = 0, we recover Q-learning algorithm.
        When eps_now = eps_next neq 0, we have SARSA algorithm
        """
        self.eps_now = eps_now
        self.eps_next = eps_next
        self.alpha = alpha
        self.gamma = gamma

    def set_num_actions(self, num_actions: int) -> None:
        self.q_function = defaultdict(partial(np.zeros, num_actions))
        self.num_actions = num_actions

    def act(self, state: StateT) -> int:

        if np.random.random() < self.eps_now:
            return np.random.randint(self.num_actions)
        return cast(int, np.argmax(self.q_function[state]))

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        if np.random.random() < self.eps_next:
            max_q = np.random.choice(self.q_function[new_state])
        else:
            max_q = np.max(self.q_function[new_state])

        self.q_function[state][action] += self.alpha * (reward + self.gamma * max_q - self.q_function[state][action])
