"""
Tests for rlmarket/environment/exchange.py
"""
from rlmarket.environment import Exchange
from rlmarket.environment.exchange_elements import Position, Imbalance
from rlmarket.market import LimitOrder, MarketOrder, DeleteOrder


anchor = 34200000000000
delta = 30000000
start_time = anchor + 9 * delta

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
    DeleteOrder(anchor + 19 * delta, 7),
    DeleteOrder(anchor + 20 * delta, 8),  # Trigger MarketOrder for liquidation
    DeleteOrder(anchor + 21 * delta, 9),
    DeleteOrder(anchor + 22 * delta, 10),
    DeleteOrder(anchor + 23 * delta, 5),
]


def test_basic_exchange(mocker):
    """
    * Test order placement with latency
        * Test that once an order is executed, another will be cancelled
    * Test liquidation
    """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=tape)
    mocker.patch('builtins.open', mocker.mock_open())
    exchange = Exchange('', start_time=start_time, indicators=[Position(), Imbalance(1, decay=0)],
                        gamma=0.95, latency=delta, num_episodes=2, order_size=50, position_limit=100)
    state = exchange.reset()
    assert exchange.book.quote == (10000, 12000)
    assert state == (0, 50 / 250)

    # Test order placement
    state, rewards, done = exchange.step(0)  # Placed order at 10000 and 12000
    assert state == (50, -75 / 125)  # Real level is already move to the 9000 level
    assert rewards is None
    assert not done

    assert exchange.total_value == -10000 * 50
    assert exchange.spread_profit == 500 * 50  # Mid price is 10500 and executed price is 10000
    assert exchange.book.bid_book.user_order_info is None
    assert exchange.book.ask_book.user_order_info is None  # Order is cancelled

    # Test order placement - complete an update full cycle to output rewards
    state, rewards, done = exchange.step(0)  # Placing order at 9000 and 12000
    assert exchange.book.quote == (9000, 13000)
    assert state == (0, -75 / 125)  # Real level is already move to the 9000 level
    assert rewards == [77500 * 0.95, 75000 + 25000 * 0.1, 7.75]  # Mid price is 11000 and executed price is 12000
    assert not done

    # Last cycle finishes. Clean start
    assert exchange.total_value == 0
    assert exchange.spread_profit == 0
    assert exchange.book.bid_book.user_order_info is None
    assert exchange.book.ask_book.user_order_info is None

    # Test liquidation
    state, rewards, done = exchange.step(0)  # Placing order at 9000 and 13000
    assert exchange.book.quote == (8000, 13000)
    assert state == (50, -50 / 150)  # Real level is at 9000 and 12000
    assert rewards is None
    assert exchange.total_value == -450000  # Long 50 shares at 9000
    assert exchange.spread_profit == 75000  # Spread between 9000 and 10500 on 50 shares
    assert not done

    state, rewards, done = exchange.step(0)  # Placing order at 8000 and 13000
    assert exchange.book.quote == (6000, 14000)  # Order 7 is cancelled
    print(exchange.book.get_depth(5))
    assert state == (80, 0)  # Liquidation. Also, Order 5 of 50 shares vs Order 8 of 50
    #   Bid     Mid     Ask     Shares  Price   Comment
    #   8000    10500   13000   -50     9000    Buy 50
    #   6000    9500    13000   -50     8000    Buy 50
    #   6000    10000   14000   20      6000    Liquidate 20
    #   6000    10000   14000   80      10000   Mark to Market
    # Total PnL is 70000
    # Spread PnL is 70000
    # Position PnL is 0. Position PnL is benchmarked to mid prices. Therefore, -50 * 10500 - 50 * 9500 + 100 * 10000 = 0
    assert rewards == [70000 * 0.95, 70000 + 0, 7]
    assert exchange.total_value == 0
    assert exchange.spread_profit == 0
    assert not done

    # Market is cleared
    state, rewards, done = exchange.step(0)
    assert state == ()
    assert rewards is None
    assert done
    assert exchange.book.empty
