"""
Tests for rlmarket/environment/exchange.py
"""
from rlmarket.environment import Exchange
from rlmarket.environment.exchange_elements import Position, Imbalance
from rlmarket.market import LimitOrder, MarketOrder, DeleteOrder


anchor = 34200000000000
delta = 30000000
start_time = anchor + 9 * delta
end_time = anchor + 21 * delta

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
    MarketOrder(anchor + 11 * delta, 2, 'S', 25),  # Order -1 executed
    MarketOrder(anchor + 12 * delta, 6, 'B', 100),
    MarketOrder(anchor + 13 * delta, 7, 'B', 100),  # Order -4 executed
    MarketOrder(anchor + 15 * delta, 2, 'S', 25),
    MarketOrder(anchor + 16 * delta, 3, 'S', 50),  # Order -5 executed
    MarketOrder(anchor + 17 * delta, 3, 'S', 50),
    MarketOrder(anchor + 18 * delta, 4, 'S', 50),  # Order -7 executed
    DeleteOrder(anchor + 19 * delta, 7),  # Liquidation
    DeleteOrder(anchor + 20 * delta, 8),
    MarketOrder(anchor + 21 * delta, 9, 'B', 50),  # Order -10 executed
    DeleteOrder(anchor + 22 * delta, 10),
    DeleteOrder(anchor + 23 * delta, 5),

    # Execution
    # Place at (10000, 12000), buy 50 @ 10000, mid price 10500 (9000, 12000)
    # Place at (9000, 12000), sell 50 @ 12000, mid price 11000 (9000, 13000)
    # Place at (9000, 13000), buy 50 @ 9000, mid price 10500 (8000, 13000)
    # Place at (8000, 13000), buy 50 @ 8000, mid price 9500 (6000, 13000)
    # Liquidate 20 @ 6000, mid price 10000 (6000, 14000)
    # Place at (6000, 14000), sell 50 @ 14000, mid price 11000 (6000, 16000)
]


def test_basic_exchange(mocker):
    """
    * Test order placement with latency
        * Test that once an order is executed, another will be cancelled
    * Test liquidation
    """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=tape)
    mocker.patch('builtins.open', mocker.mock_open())

    exchange = Exchange(files=[''], indicators=[Position(), Imbalance(1, decay=0)],
                        start_time=start_time, end_time=end_time, latency=delta,
                        order_size=50, position_limit=100)

    state = exchange.reset()
    assert exchange.state_dimension == 2
    assert exchange.book.quote == (10000, 12000)
    assert state == (0, 50 / 250)

    # Test order placement
    state, reward, metric, done = exchange.step(0)  # Placed order at 10000 and 12000
    assert state == (50, -75 / 125)  # Real level is already move to the 9000 level
    assert reward == 0
    assert metric == 0
    assert not done

    assert exchange.book.bid_book.user_order_info is None
    assert exchange.book.ask_book.user_order_info is None  # Order is cancelled

    # Test order placement - complete an update full cycle to output rewards
    state, reward, metric, done = exchange.step(0)  # Placing order at 9000 and 12000
    assert exchange.book.quote == (9000, 13000)
    assert state == (0, -75 / 125)  # Real level is already move to the 9000 level
    assert reward == 2.5 + 2.5 * 0.1  # Last reward
    assert metric == 2.5 + 2.5
    assert not done

    # Last cycle finishes. Clean start
    assert exchange.book.bid_book.user_order_info is None
    assert exchange.book.ask_book.user_order_info is None

    # Test liquidation
    state, reward, metric, done = exchange.step(0)  # Placing order at 9000 and 13000
    assert exchange.book.quote == (8000, 13000)
    assert state == (50, -50 / 150)  # Real level is at 9000 and 12000
    assert reward == 5  # No position pnl because position was 0
    assert metric == 5
    assert not done

    state, reward, metric, done = exchange.step(0)  # Placing order at 8000 and 13000
    assert exchange.book.quote == (6000, 14000)  # Order 7 is cancelled
    print(exchange.book.get_depth(5))
    assert state == (80, 0)  # Liquidation. Also, Order 5 of 50 shares vs Order 8 of 50
    assert reward == 7.5 - 5
    assert metric == 7.5 - 5
    assert not done

    # Market is cleared
    state, reward, metric, done = exchange.step(0)
    assert state == (30, 0)
    assert reward == 7.5 - 8 + 13 * 0.1
    assert metric == 7.5 - 8 + 13
    assert not done
    assert not exchange.book.empty

    # Final cleanup
    exchange.clean_up()
    assert exchange.book.empty

    # Total Profit
    #   Buy 50 @ 10000
    #   Sell 50 @ 12000
    #   Buy 50 @ 9000
    #   Buy 50 @ 8000
    #   Sell 20 @ 6000
    #   Sell 80 @ 11000
    # Total is 25 = 5 + 5 + 2.5 + 12.5
