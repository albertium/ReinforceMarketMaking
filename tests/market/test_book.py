"""
Tests for rlmarket/market/book.py
"""
import pytest

from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder
from rlmarket.market import UserLimitOrder, UserMarketOrder
from rlmarket.market.book import Book


def test_book():
    """ Test all public methods of Book """

    book = Book('B', lambda x: -x)
    assert book.side == 'B'
    assert not book.prices
    assert not book.price_levels
    assert not book.order_pool

    # Test add LimitOrder
    book.add_limit_order(LimitOrder(1, 1, 'B', 10000, 100))
    assert book.prices[0] == 10000
    assert 10000 in book.price_levels
    assert 1 in book.order_pool

    book.add_limit_order(LimitOrder(2, 2, 'B', 20000, 150))
    assert book.prices[0] == 20000
    assert book.prices[1] == 10000
    assert 20000 in book.price_levels
    assert 2 in book.order_pool

    book.add_limit_order(LimitOrder(3, 3, 'B', 20000, 100))
    assert 3 in book.order_pool

    with pytest.raises(RuntimeError, match='Market order being matched against levels not in the front'):
        book.match_limit_order(MarketOrder(4, 1, 'S', 100))

    # Test match MarketOrder
    exhausted = book.match_limit_order(MarketOrder(4, 2, 'S', 150))
    assert exhausted
    assert 10000 in book.prices
    assert 20000 in book.prices
    assert 2 not in book.order_pool

    # Same side
    with pytest.raises(RuntimeError) as error:
        _ = book.match_limit_order(MarketOrder(4, 3, 'B', 100))
    assert 'LimitOrder and MarketOrder are on the same side (B)' in str(error)

    # Match more than available
    with pytest.raises(RuntimeError, match='Market order shares 500 is more than limit order shares 100'):
        _ = book.match_limit_order(MarketOrder(4, 3, 'S', 500))

    exhausted = book.match_limit_order(MarketOrder(5, 3, 'S', 100))
    assert exhausted
    assert 20000 not in book.prices
    assert 20000 not in book.price_levels
    assert 3 not in book.order_pool

    exhausted = book.match_limit_order(MarketOrder(6, 1, 'S', 100))
    assert exhausted
    assert len(book.prices) == 0
    assert len(book.price_levels) == 0
    assert len(book.order_pool) == 0

    # Test order cancellation
    book.add_limit_order(LimitOrder(7, 4, 'B', 30000, 100))

    with pytest.raises(RuntimeError, match='Cancel more shares than available'):
        book.cancel_order(CancelOrder(8, 4, 100))

    assert len(book.prices) == 1
    assert len(book.price_levels) == 1
    assert len(book.order_pool) == 1

    # Test order deletion
    book.delete_order(DeleteOrder(9, 4))
    assert len(book.prices) == 0
    assert len(book.price_levels) == 0
    assert len(book.order_pool) == 0


def test_user_orders():
    """ Test operation with user orders """
    book = Book('B', key_func=lambda x: -x)
    assert not book.prices

    # Test add user limit orders
    book.add_user_limit_order(UserLimitOrder(1, -1, 'B', 10000, 100))
    assert -1 in book.user_order_pool
    assert not book.order_pool
    assert 10000 in book.prices
    assert 10000 in book.price_levels

    book.add_user_limit_order(UserLimitOrder(2, -2, 'B', 10000, 100))
    assert -1 in book.user_order_pool
    assert -2 not in book.user_order_pool

    book.add_user_limit_order(UserLimitOrder(3, -2, 'B', 11000, 100))
    assert -1 not in book.user_order_pool
    assert -2 in book.user_order_pool
