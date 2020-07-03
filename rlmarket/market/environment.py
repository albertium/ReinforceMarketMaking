"""
Environment follows OpenAI gym API
"""
import abc
from typing import Tuple, Any, Union


Numeric = Union[float, int]
StateT = Tuple[Numeric, ...]


class Environment(abc.ABC):
    """ Define interface for enivornment subclass """

    @abc.abstractmethod
    def reset(self) -> StateT:
        """ Reset environment to the initial state """

    @abc.abstractmethod
    def step(self, action: Any) -> Tuple[StateT, float, bool]:
        """ Take in action and return next states, rewards and flag for completion """

    @property
    @abc.abstractmethod
    def action_space(self) -> int:
        """ Return number of actions """
