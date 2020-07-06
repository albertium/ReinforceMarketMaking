"""
Start at (0, 0). Agent can choose to move in one of the 8 directions.
If ends at the (2.9, 2.9) to (3, 3) area, reward 1
else if any axis reach -3 or 3, reward -1
"""
from typing import Tuple, Deque

from rlmarket.market.environment import Environment, StateT


class LinearWalk2D(Environment):
    """ Linear walk is fitted to be between -3 and 3 for the ease of TileCodingAgent """

    dirs = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))

    def __init__(self) -> None:
        self.position: Tuple[float, float] = (0.0, 0.0)

    def reset(self) -> StateT:
        self.position = (0.0, 0.0)
        return self.position

    def step(self, action: int) -> Tuple[StateT, float, bool]:
        dx, dy = self.dirs[action]
        self.position = (self.position[0] + dx * 0.3, self.position[1] + dy * 0.3)

        if (abs(self.position[0] - 3) < 1e-10 and self.position[1] >= 2.7) \
                or (self.position[0] >= 2.7 and abs(self.position[1] - 3) < 1e-10):
            reward = 1
            done = True
        elif (abs(self.position[0] + 3) < 1e-10 and self.position[1] <= -2.7) \
                or (self.position[0] <= -2.7 and abs(self.position[1] + 3) < 1e-10):
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        self.position = (max(-3.0, min(3.0, self.position[0])), max(-3.0, min(3.0, self.position[1])))
        return self.position, reward, done

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        pass

    @property
    def action_space(self) -> int:
        return 8

    @property
    def state_dimension(self) -> int:
        return 2
