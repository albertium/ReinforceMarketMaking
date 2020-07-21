"""
Enhanced value iteration agent with memory
"""
import abc

from rlmarket.agent.value_agent import ValueIterationAgent
from rlmarket.agent.agent_elements import ReplayMemory
from rlmarket.environment import StateT


class ValueMemoryAgent(ValueIterationAgent):

    def __init__(self, eps_max: float = 1.0, eps_min: float = 0.05, warm_up_period: int = 500,
                 allow_exploration: bool = True, alpha: float = 0.1, gamma: float = 0.99,
                 memory_size: int = 5000, batch_size: int = 50) -> None:
        super().__init__(eps_max, eps_min, warm_up_period, allow_exploration, alpha, gamma)
        self.memory = ReplayMemory(memory_size=memory_size, batch_size=batch_size)
        self.minimum_size = batch_size * 10

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        self.memory.push(state, action, reward, new_state)

        if len(self.memory) < self.minimum_size:
            return

        episodes = self.memory.sample()
        for s, a, r, sp in episodes:
            super().update(s, a, r, sp)

    @abc.abstractmethod
    def initialize_q_function(self, state_dimension: int, num_actions: int) -> None:
        """ Specify q function """
