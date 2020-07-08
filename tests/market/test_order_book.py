"""
Tests for rlmarket\market\order_book.py
"""
import pytest

from rlmarket.market import OrderBook, LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder


def test_order_book():
    """ Test all public methods of OrderBook """

    # Test add order
    book = OrderBook()
    book.add_limit_order(LimitOrder(1, 1, 'B', 20000, 100))
    book.add_limit_order(LimitOrder(2, 2, 'S', 30000, 50))
    book.add_limit_order(LimitOrder(3, 3, 'S', 35000, 150))
    book.add_limit_order(LimitOrder(4, 4, 'B', 15000, 50))
    assert len(book.bid_book.order_pool) == 2
    assert len(book.ask_book.order_pool) == 2
    assert book.quote == (20000, 30000)
    assert book.get_depth() == ([(20000, 100), (15000, 50)], [(30000, 50), (35000, 150)])
    assert book.get_depth(1) == ([(20000, 100)], [(30000, 50)])

    with pytest.raises(RuntimeError, match='LimitOrder ID 1 already exists'):
        book.add_limit_order(LimitOrder(4, 1, 'B', 2500, 100))

    with pytest.raises(RuntimeError, match='Buy limit order of price 30000 cross to the ask book with quote of 30000'):
        book.add_limit_order(LimitOrder(4, 5, 'B', 30000, 100))

    with pytest.raises(RuntimeError, match='Sell limit order of price 20000 cross to the bid book with quote of 20000'):
        book.add_limit_order(LimitOrder(4, 5, 'S', 20000, 100))

    # Test match order
    book.match_limit_order(MarketOrder(5, 1, 'S', 100))
    assert len(book.bid_book.order_pool) == 1
    assert book.quote == (15000, 30000)
    assert book.get_depth() == ([(15000, 50)], [(30000, 50), (35000, 150)])

    with pytest.raises(KeyError, match='1'):
        book.match_limit_order(MarketOrder(5, 1, 'S', 100))

    with pytest.raises(RuntimeError) as error:
        book.match_limit_order(MarketOrder(5, 3, 'S', 100))
    assert 'LimitOrder and MarketOrder are on the same side (S)' in str(error)

    # Test order cancellation
    book.cancel_order(CancelOrder(6, 3, 100))
    assert book.quote == (15000, 30000)

    with pytest.raises(RuntimeError, match='Cancel more shares than available'):
        book.cancel_order(CancelOrder(6, 3, 50))

    # Test order deletion
    book.delete_order(DeleteOrder(7, 2))
    assert len(book.ask_book.order_pool) == 1
    assert book.quote == (15000, 35000)
    assert book.get_depth() == ([(15000, 50)], [(35000, 50)])

    # Test order update
    book.modify_order(UpdateOrder(8, 5, 4, 16000, 100))
    assert book.quote == (16000, 35000)
    assert book.get_depth() == ([(16000, 100)], [(35000, 50)])

    # Empty the book
    book.match_limit_order(MarketOrder(9, 5, 'S', 100))
    assert book.quote == (None, 35000)
    assert book.get_depth() == ([], [(35000, 50)])

    book.match_limit_order(MarketOrder(10, 3, 'B', 50))
    assert book.quote == (None, None)
    assert book.get_depth() == ([], [])
