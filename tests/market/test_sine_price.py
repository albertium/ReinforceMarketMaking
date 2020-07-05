import numpy as np
from numpy.testing import assert_almost_equal

from rlmarket.market import SinePrice


def test_sine_price():
    """ Test if price is generated as expected """
    env = SinePrice()
    state = env.reset()
    assert len(state) == 5
    assert state[-1] == env.level

    new_state, _, _ = env.step(0)
    assert state[1:] == new_state[:-1]

    prices = []
    for _ in range(env.cycle):
        prices.append(env.step(0)[0][-1])
    prices = np.array(prices)

    # Check number of sign changes
    signs = np.sign(prices[1:] - prices[:-1])
    sign_changes = np.sum(np.abs(np.sign(signs[1:] - signs[:-1])))

    # Length of period going up and going down should be the same within discretization error
    assert np.abs(np.sum(signs == 1) - np.sum(signs == -1)) <= 1
    assert sign_changes == 2  # Only 2 changes of signs in a cycle

    # Check peaks and trough
    assert env.level + env.amplitude * 0.95 < np.max(prices) <= env.level + env.amplitude
    assert env.level - env.amplitude * 0.95 > np.min(prices) >= env.level - env.amplitude


def test_sine_price_shift():
    """ Test other setting of price generation """
    env = SinePrice(lags=1, cycle=10)
    assert len(env.reset()) == 1

    # Check number of sign changes
    prices = np.array([env.step(0)[0][0] for _ in range(20)])
    signs = np.sign(prices[1:] - prices[:-1])
    signs = signs[signs != 0]
    sign_changes = np.sum(np.abs(np.sign(signs[1:] - signs[:-1])))
    assert sign_changes == 2 * 2  # There are 4 cycles in this period
    assert_almost_equal(prices[:10], prices[10:], decimal=10)

    # Check phase
    env = SinePrice(lags=1, cycle=20, phase=0)
    assert env.reset()[0] == env.level

    env = SinePrice(lags=1, cycle=20, phase=0.25)
    assert env.reset()[0] == env.level + env.amplitude

    env = SinePrice(lags=1, cycle=20, phase=0.5)
    assert env.reset()[0] == env.level

    env = SinePrice(lags=1, cycle=20, phase=0.75)
    assert env.reset()[0] == env.level - env.amplitude


def test_sine_price_reward():
    """ Test if environment gives correct reward """
    env = SinePrice(lags=1, cycle=20)
    env.reset()

    # First quarter is in uptrend
    for _ in range(5):
        _, reward, _ = env.step(2)
        assert reward > 0

    # Second and third quarter is in downtrend
    for _ in range(10):
        _, reward, _ = env.step(0)
        assert reward > 0

    # Last quarter is in uptrend
    for _ in range(5):
        _, reward, _ = env.step(2)
        assert reward > 0
