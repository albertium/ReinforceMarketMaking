""" Base class for Agent """
import abc

from rlmarket.market import StateT


class Agent(abc.ABC):
    """
    * Output action given states
    * Update values given rewards
    """

    @abc.abstractmethod
    def set_num_actions(self, num_actions: int) -> None:
        """ Specify number of available actions, assuming action space is discrete from 0 to N - 1 """

    @abc.abstractmethod
    def act(self, state: StateT) -> int:
        """ Given state, return the next action """

    @abc.abstractmethod
    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        """ Update value function given new state and reward """
