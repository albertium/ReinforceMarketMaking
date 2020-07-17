"""
Environment follows OpenAI gym API
"""
import abc
from typing import Tuple, Union, Deque, Optional, List


Numeric = Union[float, int]
StateT = Tuple[Numeric, ...]


class Environment(abc.ABC):
    """ Define interface for environment subclass """

    @abc.abstractmethod
    def reset(self) -> StateT:
        """ Reset environment to the initial state """

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[StateT, Optional[float], bool]:
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


class BlockEnvironment(Environment):

    episode_count: int
    reward_array: List[Tuple[float, ...]]

    def __init__(self, num_episodes: int, gamma: float) -> None:
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.decays = list(reversed([gamma ** i for i in range(num_episodes)]))

        self.reset_block()

    @abc.abstractmethod
    def reset(self) -> StateT:
        """ Total reset of environment """

    @abc.abstractmethod
    def _step(self, action: int) -> Tuple[Tuple[float, ...], bool]:
        """ Main step logic. Output reward """

    @abc.abstractmethod
    def _get_state(self) -> StateT:
        """ Return current state of environment """

    @abc.abstractmethod
    def _calculate_final_reward(self) -> float:
        """ Calculate final reward from reward array """

    @abc.abstractmethod
    def _clean_up(self) -> None:
        """ Post processing stuff like cleaning up the market """

    def reset_block(self):
        self.episode_count = 0
        self.reward_array = []  # To store cumulative reward quantities

    def step(self, action: int) -> Tuple[StateT, Optional[List[float]], Optional[float], bool]:
        self.episode_count += 1
        reward, done = self._step(action)
        self.reward_array.append(reward)

        if self.episode_count >= self.num_episodes or done:
            final_reward = self._calculate_final_reward()
            self.reset_block()

            rewards = [final_reward * decay for decay in self.decays[-self.episode_count:]]
            return self._get_state(), rewards, final_reward, done

        return self._get_state(), None, None, False
