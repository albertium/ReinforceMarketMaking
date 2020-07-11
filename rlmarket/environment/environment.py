"""
Environment follows OpenAI gym API
"""
import abc
from typing import Tuple, Union, Deque, Optional, List


Numeric = Union[float, int]
StateT = Tuple[Numeric, ...]


class Environment(abc.ABC):
    """ Define interface for environment subclass """

    def __init__(self, is_block_training: bool = False) -> None:
        # TODO: subclass for block trainer. The hint type becomes too complicated already
        self.is_block_training = is_block_training

    @abc.abstractmethod
    def reset(self) -> StateT:
        """ Reset environment to the initial state """

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[StateT, Optional[Union[float, List[float]]], bool]:
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
