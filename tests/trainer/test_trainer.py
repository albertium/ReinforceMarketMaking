"""
Unit test for rlmarket/simulator/trainer.py
"""
from typing import Tuple, Deque

from rlmarket.environment import StateT
from rlmarket.simulator import Trainer
from rlmarket.agent import Agent
from rlmarket.environment import Environment


class MockAgent(Agent):

    def __init__(self) -> None:
        self.episodes = []

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        pass

    def act(self, state: StateT) -> int:
        return 0

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        self.episodes.append((state, action, reward, new_state))

    def go_greedy(self):
        pass


class MockEnvironment(Environment):

    def __init__(self, num_rounds: int = 10) -> None:
        self.round = 0
        self.num_rounds = num_rounds

    def reset(self) -> StateT:
        return (self.round,)

    def step(self, action: int) -> Tuple[StateT, float, float, bool]:
        self.round += 1
        return (self.round,), self.round, self.round, self.round > self.num_rounds

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        pass

    def clean_up(self):
        pass

    @property
    def action_space(self) -> int:
        return 1

    @property
    def state_dimension(self) -> int:
        return 1


def test_trainer():
    """ Test basic iteration of trainer with lag """
    agent = MockAgent()
    env = MockEnvironment(num_rounds=3)
    trainer = Trainer(agent, env, lag=1)
    trainer.train(1)
    assert agent.episodes == [
        ((0,), 0, 2, (1,)),
        ((1,), 0, 3, (2,)),
        ((2,), 0, 4, (3,)),
    ]
