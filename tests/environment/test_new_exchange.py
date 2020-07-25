"""
Tests for rlmarket/environment/new_exchange.py
"""
from copy import deepcopy

from rlmarket.environment import NewExchange
from rlmarket.environment.exchange_elements import Position, Imbalance
from rlmarket.market import LimitOrder, MarketOrder, DeleteOrder, Execution


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


def test_pnl_calculation():
    """
    Test private method _calculate_pnl
    * Buy and sell
    * Same side no execution. Queue increases
    * Opposite side
        * shares < queued shares
        * shares == queued shares
        * shares > queued shares
            * Multiple shares
    """
    exchange = NewExchange(files=[''], indicators=[Position(), Imbalance(1, decay=0)],
                           reward_lb=-1000, reward_ub=1000,
                           start_time=start_time, end_time=end_time,
                           latency=delta, order_size=50, position_limit=100)

    # Case 1 - shares < queued shares
    assert exchange._calculate_pnl(Execution(1, 100, 100)) == 0
    assert exchange._calculate_pnl(Execution(2, 120, -50)) == 1000
    assert len(exchange._open_positions) == 1
    assert exchange._open_positions[0].price == 100
    assert exchange._open_positions[0].shares == 50

    # Case 2 - shares == queued shares
    assert exchange._calculate_pnl(Execution(3, 90, -50)) == -500
    assert len(exchange._open_positions) == 0

    # Case 3 - same side
    assert exchange._calculate_pnl(Execution(4, 100, -100)) == 0
    assert exchange._calculate_pnl(Execution(5, 110, -100)) == 0
    assert exchange._calculate_pnl(Execution(6, 120, -100)) == 0
    assert len(exchange._open_positions) == 3

    # Case ? - FIFO
    assert exchange._calculate_pnl(Execution(7, 110, 50)) == -500

    # Case 4 - shares > queued shares
    assert exchange._calculate_pnl(Execution(8, 90, 200)) == 10 * 50 + 20 * 100 + 30 * 50
    assert len(exchange._open_positions) == 1
    assert exchange._open_positions[0].price == 120
    assert exchange._open_positions[0].shares == -50


def test_block_exchange(mocker):
    """
    * Test if the correct reward and metric are outputted
    * Test liquidation
    """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=deepcopy(tape))
    mocker.patch('builtins.open', mocker.mock_open())

    exchange = NewExchange(files=[''], indicators=[Position(), Imbalance(1, decay=0)],
                           reward_lb=-2000, reward_ub=2000,
                           start_time=start_time, end_time=end_time,
                           latency=delta, order_size=50, position_limit=100)

    assert exchange.state_dimension == 2

    state = exchange.reset()
    assert exchange.book.quote == (10000, 12000)
    assert state == (0, 50 / 250)

    state, reward, pnl1, done = exchange.step(3)  # Buy 50 @ 10000
    assert state == (50, -75 / 125)
    assert reward == 0
    assert pnl1 == 0
    assert not done

    state, reward, pnl2, done = exchange.step(3)  # Sell 50 @ 12000
    assert state == (0, -75 / 125)
    assert reward == 10
    assert pnl2 == 10
    assert not done

    state, reward, pnl3, done = exchange.step(3)  # Buy 50 @ 9000
    assert state == (50, -50 / 150)
    assert reward == 0
    assert pnl3 == 0
    assert not done

    state, reward, pnl4, done = exchange.step(3)  # Buy 50 @ 8000 and liquidate 20 @ 6000
    assert state == (80, -50 / 150)
    assert reward == -20  # Two times of reward lower bound because of liquidation
    assert pnl4 == -6
    assert not done

    state, reward, pnl5, done = exchange.step(3)  # Sell 50 @ 13000
    assert state == (30, 25 / 75)
    assert reward == 10  # Reach upper bound
    assert pnl5 == 30 * 0.4 + 20 * 0.5
    assert not done

    state, reward, pnl6, done = exchange.step(3)  # Sell 50 @ 14000
    assert state == (-20, 0)
    assert reward == 10  # Reach upper bound
    assert pnl6 == 30 * 0.6
    assert not done

    # Total
    total_pnl = pnl1 + pnl2 + pnl3 + pnl4 + pnl5 + pnl6
    assert total_pnl == -50 * 1 + 50 * 1.2 - 50 * 0.9 - 50 * 0.8 + 20 * 0.6 + 50 * 1.3 + 30 * 1.4

    # Finishing
    assert not exchange.book.empty
    exchange.clean_up()
    assert exchange.book.empty

    # Test restart. Should output the same stats
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=deepcopy(tape))
    state = exchange.reset()
    assert exchange.book.quote == (10000, 12000)
    assert state == (0, 50 / 250)

    _, _, metric, _ = exchange.step(3)
    assert metric == 0
    _, _, metric, _ = exchange.step(3)
    assert metric == 10
    _, _, metric, _ = exchange.step(3)
    assert metric == 0
    _, _, metric, _ = exchange.step(3)
    assert metric == -6
    _, _, metric, _ = exchange.step(3)
    assert metric == 22
    _, _, metric, _ = exchange.step(3)
    assert metric == 18
