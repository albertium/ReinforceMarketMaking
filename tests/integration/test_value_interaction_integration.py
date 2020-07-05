"""
Test value iteration behaves as expected using cliff-like example
"""
from typing import Tuple

import pytest
import numpy as np

from rlmarket.agent import SimpleTDAgent
from rlmarket.market import GridWorld
from rlmarket.market.grid_world import PositionDelta
from rlmarket.simulator import Simulator


TEST_DATA = [
    pytest.param(
        # SARSA will avoid cells hear the hole
        # eps_next
        0.1,
        # nrows
        3,
        # ncols
        3,
        # start
        (0, 0),
        # hole
        (0, 1),
        # end
        (0, 2),
        # expected
        [(1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],

        id='SARSA'
    ),

    pytest.param(
        # Q-learning only care about the optimal path without "buffer
        # eps_next
        0.,
        # nrows
        3,
        # ncols
        3,
        # start
        (0, 0),
        # hole
        (0, 1),
        # end
        (0, 2),
        # expected
        [(1, 0), (1, 1), (1, 2), (0, 2)],  # e-greedy makes policy to avoid cells near the hole

        id='Q-Learning'
    )
]


# pylint: disable=too-many-arguments
@pytest.mark.parametrize('eps_next, nrows, ncols, start, hole, end, expected', TEST_DATA)
def test_integration(eps_next, nrows, ncols, start, hole, end, expected):
    """ Test if learned q-function leads to optimal path """
    agent = SimpleTDAgent(eps_next=eps_next)
    env = GridWorld(nrows=nrows, ncols=ncols, start=start, end=end, hole=hole)
    sim = Simulator(agent, env)
    sim.train(n_iters=1000)

    # pylint: disable=protected-access
    curr: Tuple[int, int] = (0, 0)
    for pos in expected:
        action: PositionDelta = env._action_map[np.argmax(agent.q_function[curr])]
        curr = (curr[0] + action.x_delta, curr[1] + action.y_delta)
        assert curr == pos
