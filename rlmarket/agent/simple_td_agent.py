"""
* Simple q function that only supports discrete state
* Q-learning or SARSA agent
"""
from collections import defaultdict
from functools import partial
from typing import Dict

import numpy as np

from rlmarket.agent.value_iteration_agent import ValueIterationAgent, QFunction
from rlmarket.market import StateT


class SimpleQTable(QFunction):
    """ Simple Q-table supports discrete states """

    def __init__(self, num_actions: int) -> None:
        self.table = defaultdict(partial(np.zeros, num_actions))

    def __getitem__(self, state: StateT) -> np.ndarray:
        return self.table[state]

    def update(self, state: StateT, action: int, value: float, alpha: float):
        values = self.table[state]
        values[action] = (1 - alpha) * values[action] + alpha * value


class SimpleTDAgent(ValueIterationAgent):
    """ Agent that uses TD to learning q-values """

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        self.q_function = SimpleQTable(num_actions)
        self.num_actions = num_actions
