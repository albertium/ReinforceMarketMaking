"""
Start at 0. Agent can choose to move either to the right or left for a random distance.
If 3 is reached or exceeded, reward 1 and the walk is done
If -3 is reached or exceeded, reward -1 and the walk is done
"""
from typing import Tuple, Deque
import random

from rlmarket.market import Environment, StateT


class LinearWalk(Environment):
    """ Linear walk is fitted to be between -3 and 3 for the ease of TileCodingAgent """

    def __init__(self) -> None:
        self.position = 0

    def reset(self) -> StateT:
        self.position = 0
        return (self.position,)

    def step(self, action: int) -> Tuple[StateT, float, bool]:
        if action == 0:
            self.position -= random.random() * 0.6  # In best case, we reach the end in 5 steps
        elif action == 1:
            self.position += random.random() * 0.6
        else:
            raise ValueError(f'Unrecognized action {action}')

        if self.position >= 3:
            reward = 1
            done = True
        elif self.position <= -3:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        self.position = max(-3, min(3, self.position))
        return (self.position,), reward, done

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        pass

    @property
    def action_space(self) -> int:
        return 2

    @property
    def state_dimension(self) -> int:
        return 1
