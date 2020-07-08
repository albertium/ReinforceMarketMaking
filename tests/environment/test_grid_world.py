"""
Test GridWorld actions and rewards
"""
from rlmarket.environment import GridWorld


def test_grid_world():
    """ Test is GridWorld position change and actions align """
    env = GridWorld()
    assert env.reset() == (0, 0)
    assert env.step(2) == ((0, 0), 0, False)  # Move to the left is the boundary
    assert env.step(3) == ((0, 0), 0, False)  # Move up is the boundary
    assert env.step(0) == ((0, 1), 0, False)
    assert env.step(0) == ((0, 2), 0, False)
    assert env.step(1) == ((1, 2), 0, False)
    assert env.step(1) == ((2, 2), 0, False)
    assert env.step(2) == ((2, 1), 0, False)
    assert env.step(1) == ((3, 1), 0, False)
    assert env.step(3) == ((2, 1), 0, False)
    assert env.step(1) == ((3, 1), 0, False)
    assert env.step(1) == ((3, 1), 0, False)  # Move down is the boundary
    assert env.step(0) == ((3, 2), 0, False)
    assert env.step(0) == ((3, 3), 1, True)
    assert env.step(2) == ((3, 2), 0, False)
    assert env.step(2) == ((3, 1), 0, False)
    assert env.step(3) == ((2, 1), 0, False)
    assert env.step(3) == ((1, 1), -1, True)
