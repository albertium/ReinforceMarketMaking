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


def test_book_properties():
    """
    Test if front price and quote price are handled properly during price level operations

    * Test quote and front price
        * Test real order first and user order next on the same new level
        * Test user order first and real order next on the same new level
        * Test user order only
    """
    # Test quote when only user order is present
    book = Book('B', key_func=lambda x: -x)
    assert book.quote is None
    assert book.front_price is None
    book.add_user_limit_order(UserLimitOrder(1, -1, 'B', 10000, 100))
    assert book.quote is None
    assert book.front_price == 10000

    # Test user order first and then real order
    book.add_limit_order(LimitOrder(2, 1, 'B', 10000, 100))
    assert book.quote == 10000
    assert book.front_price == 10000
    assert len(book.prices) == 1

    # Test real order first then user order
    book.add_limit_order(LimitOrder(3, 2, 'B', 11000, 100))
    assert book.quote == 11000
    assert book.front_price == 11000

    # Test reset
    book.add_limit_order(LimitOrder(4, 3, 'B', 12000, 100))
    print(book.prices)
    assert len(book.prices) == 3
    assert len(book.price_levels) == 3
    assert len(book.order_pool) == 3
    assert book.user_order_pool is not None

    book.reset()
    assert len(book.prices) == 0
    assert len(book.price_levels) == 0
    assert len(book.order_pool) == 0
    assert book.user_order_pool is None

    # Test properties update with regular Book operations
    book.add_limit_order((LimitOrder(1, 1, 'B', 10000, 100)))
    book.add_limit_order((LimitOrder(2, 2, 'B', 9000, 100)))
    # This is the first user order. Should not need to update original order ID
    assert book.add_user_limit_order(UserLimitOrder(3, -1, 'B', 10000, 150)) is None
    assert book.quote == 10000
    assert book.front_price == 10000
    assert book.volume == 100
    assert book.get_depth(1) == [(10000, 100)]  # User orders are not counted
    assert book.get_depth(2) == [(10000, 100), (9000, 100)]
    assert book.get_depth(2) == book.get_depth(3)  # There are only 2 levels so far

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
    assert book.volume == 50


def test_user_orders_bid_side():
    """
    Test operation with user orders
    * User order creates a new PriceLevel
        * Real order match against this level
        * User order match against this level, which should fail
    * User order falls onto existing PriceLevel
        * User order is the last in the queue
        * User order is not the last in the queue

    * Test resolve book crossing
        * Real order cannot cross real order - handled by OrderBook
        * Real order crosses user order
        * User order crosses user order - handled by OrderBook
        * User order crosses real order - handled by OrderBook

    * Test user order replacement
        * From level x to level y
        * Then from level y to level y
        * Then from level y to level y again
        * Lastly from level y to level z
    """
    book = Book('B', key_func=lambda x: -x)
    assert not book.prices

    # Test new PriceLevel
    book.add_limit_order(LimitOrder(1, 1, 'B', 10000, 100))
    assert book.add_user_limit_order(UserLimitOrder(2, -1, 'B', 9000, 100)) is None
    assert book.user_order_pool[:2] == (-1, -1)
    assert book.user_order_pool[-1].price == 9000
    assert book.user_order_pool[-1].user_orders == {-1}
    assert tuple(book.order_pool.keys()) == (1,)
    assert book.order_pool[1].price == 10000
    assert book.prices == [10000, 9000]
    assert book.price_levels[10000].price == 10000
    assert book.price_levels[9000].price == 9000

    # Test order replacement, from level x to level y
    assert book.add_user_limit_order(UserLimitOrder(3, -2, 'B', 11000, 100)) == -1
    assert book.user_order_pool[:2] == (-2, -2)
    assert book.user_order_pool[-1].price == 11000
    assert book.user_order_pool[-1].user_orders == {-2}
    assert book.prices == [11000, 10000]

    # From level y to level y
    assert book.add_user_limit_order(UserLimitOrder(4, -3, 'B', 11000, 100)) == -2
    assert book.user_order_pool[:2] == (-2, -3)
    assert book.user_order_pool[-1].price == 11000
    assert book.user_order_pool[-1].user_orders == {-2}
    assert book.prices == [11000, 10000]

    # From level y to level y again
    assert book.add_user_limit_order(UserLimitOrder(5, -4, 'B', 11000, 100)) == -3
    assert book.user_order_pool[:2] == (-2, -4)
    assert book.user_order_pool[-1].price == 11000
    assert book.user_order_pool[-1].user_orders == {-2}
    assert book.prices == [11000, 10000]

    # From level y to level z
    assert book.add_user_limit_order(UserLimitOrder(6, -5, 'B', 12000, 100)) == -4
    assert book.user_order_pool[:2] == (-5, -5)
    assert book.user_order_pool[-1].price == 12000
    assert book.user_order_pool[-1].user_orders == {-5}
    assert book.prices == [12000, 10000]
    assert set(book.price_levels.keys()) == {12000, 10000}
    assert book.quote == 10000
    assert book.front_price == 12000

    # Test real MarketOrder matching user limit order
    exhausted, executions = book.match_limit_order(MarketOrder(7, 1, 'S', 50))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -5
    assert executions[0].price == 12000
    assert executions[0].shares == 100
    assert book.user_order_pool is None
    assert book.quote == 10000
    assert book.front_price == 10000

    # Test user order on existing PriceLevel. User order on non-existing PriceLevel is tested above
    book.add_user_limit_order(UserLimitOrder(8, -6, 'B', 10000, 50))
    book.add_limit_order(LimitOrder(9, 2, 'B', 10000, 100))
    exhausted, executions = book.match_limit_order(MarketOrder(10, 1, 'S', 50))
    assert exhausted
    assert len(executions) == 0
    assert book.prices == [10000]

    with pytest.raises(RuntimeError, match='Cannot execute MarketOrder on the side that also has user LimitOrder'):
        book.match_limit_order_for_user(UserMarketOrder(11, -7, 'S', 100))

    exhausted, executions = book.match_limit_order(MarketOrder(12, 2, 'S', 50))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -6
    assert executions[0].price == 10000
    assert executions[0].shares == 50
    assert book.quote == 10000

    # Test user MarketOrder matching
    book.add_limit_order(LimitOrder(13, 3, 'B', 11000, 50))
    execution = book.match_limit_order_for_user(UserMarketOrder(14, -8, 'S', 100))
    assert execution.id == -8
    assert execution.price == 10500
    assert execution.shares == -100

    with pytest.raises(RuntimeError, match='User market order cannot be fully executed'):
        book.match_limit_order_for_user(UserMarketOrder(15, -9, 'S', 200))

    # Test book crossing - only real order crossing user order is possible. See above
    book.add_user_limit_order(UserLimitOrder(16, -10, 'B', 12000, 100))
    assert book.front_price == 12000
    assert book.quote == 11000

    with pytest.raises(RuntimeError, match='Real order crosses real order'):
        book.resolve_book_crossing_on_user_orders(10500)

    executions = book.resolve_book_crossing_on_user_orders(11500)
    assert len(executions) == 1
    assert executions[0].id == -10
    assert executions[0].price == 12000
    assert executions[0].shares == 100
    assert book.front_price == 11000
    assert book.quote == 11000


def test_user_orders_ask_side():
    """ Test operation with user orders on ask side. See the bid side for description """
    book = Book('S', key_func=None)
    assert not book.prices

    # Test new PriceLevel
    book.add_limit_order(LimitOrder(1, 1, 'S', 10000, 100))
    assert book.add_user_limit_order(UserLimitOrder(2, -1, 'S', 11000, 100)) is None
    assert book.user_order_pool[:2] == (-1, -1)
    assert book.user_order_pool[-1].price == 11000
    assert book.user_order_pool[-1].user_orders == {-1}
    assert tuple(book.order_pool.keys()) == (1,)
    assert book.order_pool[1].price == 10000
    assert book.prices == [10000, 11000]
    assert book.price_levels[10000].price == 10000
    assert book.price_levels[11000].price == 11000

    # Test order replacement, from level x to level y
    assert book.add_user_limit_order(UserLimitOrder(3, -2, 'S', 9000, 100)) == -1
    assert book.user_order_pool[:2] == (-2, -2)
    assert book.user_order_pool[-1].price == 9000
    assert book.user_order_pool[-1].user_orders == {-2}
    assert book.prices == [9000, 10000]

    # From level y to level y
    assert book.add_user_limit_order(UserLimitOrder(4, -3, 'S', 9000, 100)) == -2
    assert book.user_order_pool[:2] == (-2, -3)
    assert book.user_order_pool[-1].price == 9000
    assert book.user_order_pool[-1].user_orders == {-2}
    assert book.prices == [9000, 10000]

    # From level y to level y again
    assert book.add_user_limit_order(UserLimitOrder(5, -4, 'S', 9000, 100)) == -3
    assert book.user_order_pool[:2] == (-2, -4)
    assert book.user_order_pool[-1].price == 9000
    assert book.user_order_pool[-1].user_orders == {-2}
    assert book.prices == [9000, 10000]

    # From level y to level z
    assert book.add_user_limit_order(UserLimitOrder(6, -5, 'S', 9500, 100)) == -4
    assert book.user_order_pool[:2] == (-5, -5)
    assert book.user_order_pool[-1].price == 9500
    assert book.user_order_pool[-1].user_orders == {-5}
    assert book.prices == [9500, 10000]
    assert set(book.price_levels.keys()) == {9500, 10000}
    assert book.quote == 10000
    assert book.front_price == 9500

    # Test real MarketOrder matching user limit order
    exhausted, executions = book.match_limit_order(MarketOrder(7, 1, 'B', 50))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -5
    assert executions[0].price == 9500
    assert executions[0].shares == -100
    assert book.user_order_pool is None
    assert book.quote == 10000
    assert book.front_price == 10000

    # Test user order on existing PriceLevel. User order on non-existing PriceLevel is tested above
    book.add_user_limit_order(UserLimitOrder(8, -6, 'S', 10000, 50))
    book.add_limit_order(LimitOrder(9, 2, 'S', 10000, 100))
    exhausted, executions = book.match_limit_order(MarketOrder(10, 1, 'B', 50))
    assert exhausted
    assert len(executions) == 0
    assert book.prices == [10000]

    with pytest.raises(RuntimeError, match='Cannot execute MarketOrder on the side that also has user LimitOrder'):
        book.match_limit_order_for_user(UserMarketOrder(11, -7, 'S', 100))

    exhausted, executions = book.match_limit_order(MarketOrder(12, 2, 'B', 50))
    assert not exhausted
    assert len(executions) == 1
    assert executions[0].id == -6
    assert executions[0].price == 10000
    assert executions[0].shares == -50
    assert book.quote == 10000

    # Test user MarketOrder matching
    book.add_limit_order(LimitOrder(13, 3, 'B', 11000, 50))
    execution = book.match_limit_order_for_user(UserMarketOrder(14, -8, 'S', 100))
    assert execution.id == -8
    assert execution.price == 10500
    assert execution.shares == -100

    with pytest.raises(RuntimeError, match='User market order cannot be fully executed'):
        book.match_limit_order_for_user(UserMarketOrder(15, -9, 'S', 200))

    # Test book crossing - only real order crossing user order is possible. See above
    book.add_user_limit_order(UserLimitOrder(16, -10, 'S', 8000, 100))
    assert book.front_price == 8000
    assert book.quote == 10000

    with pytest.raises(RuntimeError, match='Real order crosses real order'):
        book.resolve_book_crossing_on_user_orders(10500)

    executions = book.resolve_book_crossing_on_user_orders(9000)
    assert len(executions) == 1
    assert executions[0].id == -10
    assert executions[0].price == 8000
    assert executions[0].shares == -100
    assert book.front_price == 10000
    assert book.quote == 10000


def test_crossing_handling():
    """ Test if crossing handling still works when there is only user order """
    book = Book('B', lambda x: -x)
    assert book.add_user_limit_order(UserLimitOrder(1, -1, 'B', 10000, 100)) is None
    assert book.front_price == 10000
    executions = book.resolve_book_crossing_on_user_orders(10000)
    assert book.front_price is None
    assert len(executions) == 1
    assert executions[0].id == -1
    assert executions[0].price == 10000
    assert executions[0].shares == 100
