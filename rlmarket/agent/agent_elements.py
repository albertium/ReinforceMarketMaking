from typing import Tuple, Deque, List
import random
from collections import deque

from rlmarket.environment import StateT


class ReplayMemory:
    """ Store episodes for DQN training """

    def __init__(self, memory_size: int, batch_size: int):
        self.batch_size = batch_size
        self.memory: Deque[Tuple[StateT, int, float, StateT]] = deque(maxlen=memory_size)

    def push(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        """ Saves a transition. """
        self.memory.append((state, action, reward, new_state))

    def sample(self) -> List[Tuple[StateT, int, float, StateT]]:
        """ Sample past episodes for training """
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
