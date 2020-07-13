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
    exhausted, _ = book.match_limit_order(MarketOrder(4, 2, 'S', 150))
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

    exhausted, _ = book.match_limit_order(MarketOrder(5, 3, 'S', 100))
    assert exhausted
    assert 20000 not in book.prices
    assert 20000 not in book.price_levels
    assert 3 not in book.order_pool

    exhausted, _ = book.match_limit_order(MarketOrder(6, 1, 'S', 100))
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
    assert book.add_user_limit_order(UserLimitOrder(1, -1, 'B', 10000, 100))
    assert -1 in book.user_order_pool
    assert not book.order_pool
    assert tuple(book.prices) == (10000,)
    assert 10000 in book.price_levels

    # Should not replace existing order
    assert not book.add_user_limit_order(UserLimitOrder(2, -2, 'B', 10000, 100))
    assert -1 in book.user_order_pool
    assert -2 not in book.user_order_pool

    # Should replace existing order because price is different
    assert book.add_user_limit_order(UserLimitOrder(3, -2, 'B', 11000, 100))
    assert -1 not in book.user_order_pool
    assert -2 in book.user_order_pool
    assert tuple(book.prices) == (11000,)
    assert 10000 not in book.price_levels
    assert 11000 in book.price_levels

    # Test add order with real orders
    book.add_limit_order(LimitOrder(4, 1, 'B', 11000, 50))

    with pytest.raises(RuntimeError, match='Cannot execute MarketOrder on the side that also has user LimitOrder'):
        book.match_limit_order_for_user(UserMarketOrder(5, -3, 'S', 100))

    # When user order is front of real order
    exhausted, executions = book.match_limit_order(MarketOrder(6, 1, 'S', 50))
    assert len(executions) == 1
    assert executions[0].id == -2
    assert executions[0].price == 11000
    assert executions[0].shares == 100
    assert len(book.prices) == 0
    assert len(book.price_levels) == 0

    # When real order is in front of user order
    book.add_limit_order(LimitOrder(7, 2, 'B', 12000, 100))
    assert book.add_user_limit_order(UserLimitOrder(8, -4, 'B', 12000, 50))
    exhausted, executions = book.match_limit_order(MarketOrder(9, 2, 'S', 100))
    assert exhausted
    assert len(executions) == 0
    assert tuple(book.prices) == (12000,)

    # Add a real order below the current level so to test if the user order is executed
    book.add_limit_order(LimitOrder(10, 3, 'B', 11000, 100))
    book.add_limit_order(LimitOrder(11, 4, 'B', 10000, 100))
    assert tuple(book.prices) == (12000, 11000, 10000)

    exhausted, executions = book.match_limit_order(MarketOrder(13, 3, 'S', 50))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -4
    assert executions[0].price == 12000
    assert executions[0].shares == 50
    assert tuple(book.prices) == (11000, 10000)

    # Test user market order
    execution = book.match_limit_order_for_user(UserMarketOrder(14, -5, 'S', 100))
    assert execution.id == -5
    assert execution.price == 10500
    assert execution.shares == -100
    # User order won't actually consume real orders
    assert tuple(book.prices) == (11000, 10000)

    with pytest.raises(RuntimeError, match='User market order cannot be fully executed'):
        book.match_limit_order_for_user(UserMarketOrder(15, -6, 'S', 200))


def test_user_order_ask_book():
    """ Supplement test with ask book """

    # Test user limit order
    book = Book('S', key_func=None)
    assert book.add_user_limit_order(UserLimitOrder(1, -1, 'S', 12000, 100))
    book.add_limit_order(LimitOrder(2, 1, 'S', 12000, 100))
    book.add_limit_order(LimitOrder(3, 2, 'S', 13000, 100))
    assert tuple(book.prices) == (12000, 13000)
    exhausted, executions = book.match_limit_order(MarketOrder(4, 1, 'B', 50))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -1
    assert executions[0].price == 12000
    assert executions[0].shares == -100

    # Test user market order
    book.add_limit_order(LimitOrder(5, 3, 'S', 15000, 200))
    assert tuple(book.prices) == (12000, 13000, 15000)

    execution = book.match_limit_order_for_user(UserMarketOrder(6, -2, 'B', 350))
    assert execution.id == -2
    assert execution.price == 14000
    assert execution.shares == 350

    with pytest.raises(RuntimeError, match='User market order cannot be fully executed'):
        book.match_limit_order_for_user(UserMarketOrder(7, -3, 'B', 351))


def test_front_and_quote_price():
    """ Test if front price and quote price are handled properly during price level operations """
    book = Book('B', key_func=lambda x: -x)
    book.add_limit_order((LimitOrder(1, 1, 'B', 10000, 100)))
    book.add_limit_order((LimitOrder(2, 2, 'B', 9000, 100)))
    assert book.add_user_limit_order(UserLimitOrder(3, -1, 'B', 10000, 100))
    assert book.quote == 10000
    assert book.front_price == 10000

    # Real order is exhausted but user order is still there
    exhausted, executions = book.match_limit_order(MarketOrder(4, 1, 'S', 100))
    assert exhausted
    assert len(executions) == 0
    assert book.quote == 9000
    assert book.front_price == 10000

    # This time the user order is matched and the 10000 level is completely gone
    exhausted, executions = book.match_limit_order(MarketOrder(5, 2, 'S', 50))
    assert not exhausted
    assert len(executions) == 1
    assert book.quote == 9000
    assert book.front_price == 9000


def test_crossing_handling():
    """ Test crossing """

    # Test book statistics
    # TODO: We only allow one user order currently
    book = Book('S', key_func=None)
    assert book.add_user_limit_order(UserLimitOrder(1, -1, 'S', 10000, 100))
    book.add_limit_order(LimitOrder(4, 1, 'S', 12000, 100))

    assert book.front_price == 10000
    assert book.quote == 12000

    # Test crossing
    executions = book.resolve_limit_order_crossing(10000)
    assert len(book.user_order_pool) == 0
    assert book.front_price == 12000
    assert book.quote == 12000
    assert len(executions) == 1
    assert executions[0].id == -1
    assert executions[0].price == 10000
    assert executions[0].shares == 100


def test_crossing_handling2():
    """ Test crossing when real order is added first """
    book = Book('B', key_func=lambda x: -x)
    book.add_limit_order(LimitOrder(1, 1, 'B', 10000, 100))
    assert book.add_user_limit_order(UserLimitOrder(2, -1, 'B', 11000, 100))
    assert book.quote == 10000
    assert book.front_price == 11000

    book.add_limit_order(LimitOrder(3, 2, 'B', 9000, 100))
    assert book.quote == 10000
    assert book.front_price == 11000

    book.add_limit_order(LimitOrder(4, 3, 'B', 12000, 100))
    assert book.quote == 12000
    assert book.front_price == 12000

    book.delete_order(DeleteOrder(5, 3))
    assert book.quote == 10000
    assert book.front_price == 11000

    book.delete_order(DeleteOrder(6, 1))
    assert book.quote == 9000
    assert book.front_price == 11000

    exhausted, executions = book.match_limit_order(MarketOrder(7, 2, 'S', 50))
    assert book.quote == 9000
    assert book.front_price == 9000
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -1
    assert executions[0].price == 11000
    assert executions[0].shares == 100

    # Add user order in front of real order
    assert book.add_user_limit_order(UserLimitOrder(8, -2, 'B', 10000, 100))
    assert book.quote == 9000
    assert book.front_price == 10000

    exhausted, executions = book.match_limit_order(MarketOrder(9, 2, 'S', 25))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -2
    assert executions[0].shares == 100
    assert book.quote == 9000
    assert book.front_price == 9000

    # Add user order at the back of real order
    assert book.add_user_limit_order(UserLimitOrder(10, -3, 'B', 8000, 100))
    exhausted, executions = book.match_limit_order(MarketOrder(11, 2, 'S', 25))
    assert exhausted
    assert len(executions) == 0
    assert book.quote is None
    assert book.front_price == 8000


def test_crossing_handling3():
    """ Test if crossing handling still works when there is only user order """
    book = Book('B', lambda x: -x)
    assert book.add_user_limit_order(UserLimitOrder(1, -1, 'B', 10000, 100))
    assert book.front_price == 10000
    executions = book.resolve_limit_order_crossing(10000)
    assert book.front_price is None
    assert len(executions) == 1
    assert executions[0].id == -1
    assert executions[0].price == 10000
    assert executions[0].shares == 100
