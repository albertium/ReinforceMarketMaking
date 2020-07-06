"""
Environment follows OpenAI gym API
"""
import abc
from typing import Tuple, Union, Deque


Numeric = Union[float, int]
StateT = Tuple[Numeric, ...]


class Environment(abc.ABC):
    """ Define interface for enivornment subclass """

    @abc.abstractmethod
    def reset(self) -> StateT:
        """ Reset environment to the initial state """

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[StateT, float, bool]:
        """ Take in action and return next states, rewards and flag for completion """

    @abc.abstractmethod
    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        """ Take in episode memory and render visual display accordingly """

    @property
    @abc.abstractmethod
    def action_space(self) -> int:
        """ Return number of actions """

    @property
    @abc.abstractmethod
    def state_dimension(self) -> int:
        """ Return state dimension """
