"""
Basic tests for 2D Linear Walk
Linear Walk is used for testing tile coding
"""
from numpy.testing import assert_almost_equal

from rlmarket.market.linear_walk_2d import LinearWalk2D


def test_linear_walk_2d():
    """ Test if we can reach the final point """
    env = LinearWalk2D()
    for dist in [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]:
        state, reward, done = env.step(0)
        assert_almost_equal(state, (dist, dist), decimal=10)
        assert reward == 0
        assert not done

    state, reward, done = env.step(0)
    assert_almost_equal(state, (3.0, 3.0), decimal=10)
    assert reward == 1
    assert done
