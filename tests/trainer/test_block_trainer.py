"""
Unit test for rlmarket/simulator/block_trainer.py
"""
from typing import Tuple, Deque

from rlmarket.agent import Agent
from rlmarket.environment import Environment, StateT
from rlmarket.simulator import BlockTrainer


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

    def go_normal(self):
        pass


class MockEnvironment(Environment):
    """ Mock a block environment """

    def __init__(self, num_rounds: int, block_size: int) -> None:
        self.num_rounds = num_rounds
        self.block_size = block_size + 1
        self.round = 0

    def reset(self) -> StateT:
        return (self.round,)

    def step(self, action: int) -> Tuple[StateT, float, float, bool]:
        self.round += 1
        reward = self.round if self.round % self.block_size == 0 else None
        return (self.round,), reward, reward, self.round >= self.num_rounds

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
    """ Test if block trainer chunk out the correct episodes """
    agent = MockAgent()
    env = MockEnvironment(num_rounds=12, block_size=3)
    trainer = BlockTrainer(agent, env, block_size=3, gamma=0.5)
    trainer.train(1)
    assert agent.episodes == [
        ((0,), 0, 1.0, (1,)),
        ((1,), 0, 2.0, (2,)),
        ((2,), 0, 4.0, (3,)),

        ((4,), 0, 2.0, (5,)),
        ((5,), 0, 4.0, (6,)),
        ((6,), 0, 8.0, (7,)),

        ((8,), 0, 3.0, (9,)),
        ((9,), 0, 6.0, (10,)),
        ((10,), 0, 12.0, (11,)),
    ]

    # Test different number of rounds
    agent = MockAgent()
    env = MockEnvironment(num_rounds=11, block_size=3)
    trainer = BlockTrainer(agent, env, block_size=3, gamma=0.5)
    trainer.train(1)
    assert agent.episodes == [
        ((0,), 0, 1.0, (1,)),
        ((1,), 0, 2.0, (2,)),
        ((2,), 0, 4.0, (3,)),

        ((4,), 0, 2.0, (5,)),
        ((5,), 0, 4.0, (6,)),
        ((6,), 0, 8.0, (7,)),
    ]
