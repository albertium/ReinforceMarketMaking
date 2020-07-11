"""
Tests for rlmarket/market/price_level.py
"""
import pytest

from rlmarket.market.price_level import PriceLevel
from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder
from rlmarket.market import UserLimitOrder, UserMarketOrder


def test_price_level():
    """ Test public methods of PriceLevel """

    price_level = PriceLevel(10000)
    assert price_level.price == 10000

    # First order
    price_level.add_limit_order(LimitOrder(1, 1, 'B', 10000, 100))
    assert price_level.shares == 100
    assert price_level.length == 1
    assert 1 in price_level.queue

    # Second order
    price_level.add_limit_order(LimitOrder(2, 2, 'B', 10000, 200))
    assert price_level.shares == 300
    assert price_level.length == 2
    assert 2 in price_level.queue
    order_iter = iter(price_level.queue.keys())

    # Test ordering
    assert next(order_iter) == 1
    assert next(order_iter) == 2

    # Test wrong price
    with pytest.raises(RuntimeError):
        price_level.add_limit_order(LimitOrder(2, 2, 'B', 20000, 500))

    # Test MarketOrder matching
    _, exhausted, _ = price_level.match_limit_order(MarketOrder(3, 1, 'S', 100))
    assert exhausted
    assert price_level.shares == 200
    assert price_level.length == 1

    # Test partial match
    _, exhausted, _ = price_level.match_limit_order(MarketOrder(4, 2, 'S', 100))
    assert not exhausted
    assert price_level.shares == 100
    assert price_level.length == 1

    # Same side
    with pytest.raises(RuntimeError):
        _ = price_level.match_limit_order(MarketOrder(4, 2, 'B', 100))

    with pytest.raises(RuntimeError, match='Market order shares 500 is more than limit order shares 100'):
        _ = price_level.match_limit_order(MarketOrder(4, 2, 'S', 500))

    # Match non-exist LimitOrder
    with pytest.raises(KeyError):
        _ = price_level.match_limit_order(MarketOrder(3, 1, 'B', 100))

    # Test order cancellation
    price_level.cancel_order(CancelOrder(5, 2, 50))
    assert price_level.shares == 50
    assert price_level.queue[2].shares == 50

    # Cancel more than available
    with pytest.raises(RuntimeError):
        price_level.cancel_order(CancelOrder(5, 2, 150))

    # Test order deletion
    price_level.delete_order(DeleteOrder(6, 2))
    assert price_level.shares == 0
    assert price_level.length == 0


def test_user_order():
    """ Test PriceLevel's handling of user orders """

    price_level = PriceLevel(10000)

    # User order at the front
    price_level.add_user_limit_order(UserLimitOrder(1, -1, 'B', 10000, 100))
    assert price_level.shares == 0
    assert price_level.length == 1
    assert price_level.num_user_orders == 1
    assert -1 in price_level.user_orders

    price_level.add_limit_order(LimitOrder(2, 1, 'B', 10000, 100))
    assert price_level.length == 2

    _, exhausted, user_orders = price_level.match_limit_order(MarketOrder(3, 1, 'S', 50))
    assert not exhausted
    assert len(user_orders) == 1
    assert price_level.shares == 50
    assert price_level.length == 1
    assert price_level.num_user_orders == 0

    # User order at the back
    price_level.add_user_limit_order(UserLimitOrder(3, -2, 'B', 10000, 50))
    _, exhausted, user_orders = price_level.match_limit_order(MarketOrder(4, 1, 'S', 50))
    assert price_level.shares == 0
    assert exhausted
    assert len(user_orders) == 0
    assert price_level.length == 1
    assert price_level.num_user_orders == 1
    assert -2 in price_level.user_orders

    with pytest.raises(ValueError, match='User LimitOrder price 11000 is not the same as PriceLevel 10000'):
        price_level.add_user_limit_order(UserLimitOrder(4, -3, 'B', 11000, 100))

    with pytest.raises(ValueError, match='User order should have negative order ID'):
        price_level.add_user_limit_order(UserLimitOrder(4, 3, 'B', 11000, 100))

    # Test user MarketOrder
    with pytest.raises(RuntimeError,
                       match='User market order cannot match against price level that has user limit order'):
        price_level.match_limit_order_for_user(UserMarketOrder(5, -4, 'S', 30))

    price_level.delete_user_order(-2)
    assert price_level.length == 0
    price_level.add_limit_order(LimitOrder(5, 2, 'B', 10000, 100))
    price_level.add_limit_order(LimitOrder(6, 3, 'B', 10000, 50))
    assert price_level.match_limit_order_for_user(UserMarketOrder(7, -4, 'S', 30)) == 0
    assert price_level.shares == 150
    assert price_level.match_limit_order_for_user(UserMarketOrder(8, -5, 'S', 180)) == 30
    assert price_level.num_user_orders == 0
    assert len(price_level.user_orders) == 0
