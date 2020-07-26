"""
Unittest for rlmarket/gym_env/absolute_exchange.py
"""
from copy import deepcopy
from numpy.testing import assert_almost_equal

from rlmarket.gym_env import AbsoluteExchange
from rlmarket.environment.exchange_elements import Position, Imbalance, NormalizedPosition
from rlmarket.market import LimitOrder, MarketOrder, DeleteOrder


anchor = 34200000000000
delta = 30000000
start_time = anchor + 9 * delta
end_time = anchor + 23 * delta

tape = [
    # 5 Bid Levels
    LimitOrder(anchor + 0 * delta, 1, 'B', 10000, 150),
    LimitOrder(anchor + 1 * delta, 2, 'B', 9000, 50),
    LimitOrder(anchor + 2 * delta, 3, 'B', 8000, 100),
    LimitOrder(anchor + 3 * delta, 4, 'B', 7000, 50),
    LimitOrder(anchor + 4 * delta, 5, 'B', 6000, 50),

    # 5 Ask Levels. Spread is therefore 2000
    LimitOrder(anchor + 5 * delta, 6, 'S', 12000, 100),
    LimitOrder(anchor + 6 * delta, 7, 'S', 13000, 200),
    LimitOrder(anchor + 7 * delta, 8, 'S', 14000, 50),
    LimitOrder(anchor + 8 * delta, 9, 'S', 15000, 50),
    LimitOrder(anchor + 9 * delta, 10, 'S', 16000, 50),

    # Pre market done
    MarketOrder(anchor + 10 * delta, 1, 'S', 150),
    MarketOrder(anchor + 11 * delta, 2, 'S', 25),  # Buy order executed - 50 shares
    MarketOrder(anchor + 12 * delta, 6, 'B', 100),
    MarketOrder(anchor + 13 * delta, 7, 'B', 100),  # Sell order executed - 50 shares
    MarketOrder(anchor + 15 * delta, 2, 'S', 25),
    MarketOrder(anchor + 16 * delta, 3, 'S', 50),  # Buy order executed - 50 shares - end of Block 1
    MarketOrder(anchor + 17 * delta, 3, 'S', 50),
    MarketOrder(anchor + 18 * delta, 4, 'S', 25),  # Buy order executed - 100 shares
    MarketOrder(anchor + 19 * delta, 4, 'S', 25),  # Liquidation - 80 shares
    MarketOrder(anchor + 20 * delta, 7, 'B', 100),
    MarketOrder(anchor + 21 * delta, 8, 'B', 25),  # Sell order executed - 30 shares
    MarketOrder(anchor + 22 * delta, 8, 'B', 25),
    MarketOrder(anchor + 23 * delta, 9, 'B', 50),  # Sell order executed - -20 shares - end of Block 2
    DeleteOrder(anchor + 24 * delta, 5),
    DeleteOrder(anchor + 25 * delta, 10),

    # Execution
    # Place at (10000, 12000), buy 50 @ 10000, mid price 10500 (9000, 12000)
    # Place at (9000, 12000), sell 50 @ 12000, mid price 11000 (9000, 13000)
    # Place at (9000, 13000), buy 50 @ 9000, mid price 10500 (8000, 13000) - Block 1
    # Place at (8000, 13000), buy 50 @ 8000, mid price 10000 (7000, 13000)
    # Liquidate 20 @ 6000, mid price 9500 (6000, 13000)
    # Place at (6000, 13000), sell 50 @ 13000, mid price 10000 (6000, 14000)
    # Place at (6000, 14000), sell 50 @ 14000, mid price 11000 (6000, 16000) - Block 2
]


def test_exchange(mocker):
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=deepcopy(tape))
    mocker.patch('builtins.open', mocker.mock_open())

    position_limit = 100

    exchange = AbsoluteExchange(files=[''], indicators=[Position(), Imbalance(1, decay=0)],
                                reward_lb=-0.2, reward_ub=0.3,  # Bounds in percentage
                                start_time=start_time, end_time=end_time,
                                latency=delta, order_size=50, position_limit=position_limit)

    assert exchange.observation_space.shape == (2,)

    state = exchange.reset()
    assert exchange.book.quote == (10000, 12000)
    assert_almost_equal(state, (0, 50 / 250))

    total_pnl = 0

    state, reward, done, info = exchange.step(3)  # Buy 50 @ 10000
    assert_almost_equal(state, (50, -75 / 125))
    assert reward == 0
    assert info['pnl'] == 0
    assert not done
    total_pnl += info['pnl']

    state, reward, done, info = exchange.step(3)  # Sell 50 @ 12000
    assert_almost_equal(state, (0, -75 / 125))
    assert reward == 10
    assert info['pnl'] == 10
    assert not done
    total_pnl += info['pnl']

    state, reward, done, info = exchange.step(3)  # Buy 50 @ 9000
    assert_almost_equal(state, (50, -50 / 150))
    assert reward == 0
    assert info['pnl'] == 0
    assert not done
    total_pnl += info['pnl']

    state, reward, done, info = exchange.step(3)  # Buy 50 @ 8000 and liquidate 20 @ 6000
    assert_almost_equal(state, (80, -50 / 150))
    assert reward == -4  # Two times of reward lower bound because of liquidation
    assert info['pnl'] == -6
    assert not done
    total_pnl += info['pnl']

    state, reward, done, info = exchange.step(3)  # Sell 50 @ 13000
    assert_almost_equal(state, (30, 25 / 75))
    assert reward == 15  # Reach upper bound
    assert info['pnl'] == 30 * 0.4 + 20 * 0.5
    assert not done
    total_pnl += info['pnl']

    state, reward, done, info = exchange.step(3)  # Sell 50 @ 14000
    assert_almost_equal(state, (-20, 0))
    assert reward == 9  # Close out 30 shares at a profit of 0.36 -> 0.3 * 30 = 9
    assert info['pnl'] == 30 * 0.6
    assert not done
    total_pnl += info['pnl']

    # Total
    assert total_pnl == -50 * 1 + 50 * 1.2 - 50 * 0.9 - 50 * 0.8 + 20 * 0.6 + 50 * 1.3 + 30 * 1.4

    # Finishing
    assert not exchange.book.empty
    exchange.clean_up()
    assert exchange.book.empty

    # Test restart. Should output the same stats
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=deepcopy(tape))
    state = exchange.reset()
    assert exchange.book.quote == (10000, 12000)
    assert_almost_equal(state, (0, 50 / 250))

    _, _, _, info = exchange.step(3)
    assert info['pnl'] == 0
    _, _, _, info = exchange.step(3)
    assert info['pnl'] == 10
    _, _, _, info = exchange.step(3)
    assert info['pnl'] == 0
    _, _, _, info = exchange.step(3)
    assert info['pnl'] == -6
    _, _, _, info = exchange.step(3)
    assert info['pnl'] == 22
    _, _, _, info = exchange.step(3)
    assert info['pnl'] == 18
