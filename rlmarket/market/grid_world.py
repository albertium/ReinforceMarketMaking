"""
GridWorld simulation environment
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Deque

from rlmarket.market import Environment
from rlmarket.market.environment import StateT


class Position:
    """ Represent the location of the agent """

    # pylint: disable=invalid-name
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __ge__(self, other: Position):
        return self.x >= other.x and self.y >= other.y

    def __le__(self, other: Position):
        return self.x <= other.x and self.y <= other.y

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return NotImplemented

    def __add__(self, other: PositionDelta):
        return Position(self.x + other.x_delta, self.y + other.y_delta)

    def within(self, boundary: Boundary):
        """ Check if position is within the given boundary """
        return boundary.top_left <= self <= boundary.bottom_right

    def copy(self):
        """ Deep copy of position """
        return Position(self.x, self.y)


@dataclass
class PositionDelta:
    """ Represent the movement of agent """
    x_delta: int
    y_delta: int


@dataclass
class Boundary:
    """ Defines the bottom right boundary of GridWorld. Top left boundary is always (0, 0) """
    top_left: Position
    bottom_right: Position


class GridWorld(Environment):
    """
    A 2D GridWorld has 3 anchors. Start, hole and end.
    * Agent starts at "Start"
    * GridWorld ends when agent enters "End" which gives 1 point
    * GridWorld also ends when agent steps onto the "Hole" which gives -1 point
    """

    _action_map = {
        0: PositionDelta(0, 1),
        1: PositionDelta(1, 0),
        2: PositionDelta(0, -1),
        3: PositionDelta(-1, 0),
    }

    def __init__(self, nrows: int = 4, ncols: int = 4,
                 start: Tuple[int, int] = (0, 0), end: Tuple[int, int] = (3, 3),
                 hole: Tuple[int, int] = (1, 1)) -> None:

        self.boundary = Boundary(Position(0, 0), Position(nrows - 1, ncols - 1))
        self.start = Position(*start)
        self.end = Position(*end)
        self.hole = Position(*hole)
        self.state = Position(0, 0)

        if not self.start.within(self.boundary) \
                or not self.end.within(self.boundary) \
                or not self.hole.within(self.boundary):
            raise ValueError('Out of boundary')

    def calculate_reward(self) -> int:
        """ Return corresponding reward of the current position """
        if self.state == self.end:
            return 1
        if self.state == self.hole:
            return -1
        return 0

    def reset(self) -> StateT:
        self.state = self.start.copy()
        return self.state.x, self.state.y

    def step(self, action: int) -> Tuple[StateT, float, bool]:
        delta = self._action_map.get(action, None)
        if not delta:
            raise ValueError(f'Invalid action: {action}')

        new_position = self.state + delta
        if new_position.within(self.boundary):
            self.state = new_position
            reward = self.calculate_reward()
            return (self.state.x, self.state.y), float(reward), reward != 0
        return (self.state.x, self.state.y), 0, False

    @property
    def action_space(self) -> int:
        return 4

    @property
    def state_dimension(self) -> int:
        return 2

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        pass
