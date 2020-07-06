"""
Classic cliff walking problem.
* Q-learning will choose the shortest road
* SARSA will pick the longer road away from the cliff
"""
from typing import Tuple, Deque
import numpy as np

from rlmarket.market import Environment, StateT


class Cliff(Environment):

    directions = 0, 1, 0, -1, 0
    n_rows = 4
    n_cols = 10
    position: Tuple[int, int]

    def reset(self) -> StateT:
        self.position = 0, 0
        return self.position

    def step(self, action: int) -> Tuple[StateT, float, bool]:
        dir1, dir2 = self.directions[action], self.directions[action + 1]
        new_position = self.position[0] + dir1, self.position[1] + dir2
        if 0 <= new_position[0] < self.n_rows and 0 <= new_position[1] < self.n_cols:
            self.position = new_position
            if self.position[0] == 0 and 1 <= self.position[1] < self.n_cols - 1:  # cliff
                reward = -100
                done = True
            elif self.position[0] == 0 and self.position[1] == self.n_cols - 1:
                reward = 1
                done = True
            else:
                reward = -1
                done = False

            return self.position, reward, done

        return self.position, -1, False

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        board[0, 1: -1] = -100
        for idx, (state, _, _, _) in enumerate(memory):
            board[state] = idx + 1
        if len(memory):
            board[memory[-1][-1]] = len(memory) + 1
        print(board)

    @property
    def action_space(self) -> int:
        return 4

    @property
    def state_dimension(self) -> int:
        return 2
