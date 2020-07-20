"""
Test for Tape at rlmarket/environment/exchange_elements.py
"""
import numpy as np

from rlmarket.environment.exchange_elements import Tape, MidPriceDeltaSign, Imbalance, Position
from rlmarket.environment import Exchange
from rlmarket.market import LimitOrder, MarketOrder, UserLimitOrder


def test_tape(mocker):
    """ Test all public methods of Tape """
    messages = [
        LimitOrder(1, 1, 'B', 10000, 100),
        LimitOrder(2, 2, 'B', 10000, 100),
        LimitOrder(3, 3, 'B', 10000, 100),
    ]

    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=messages)
    mocker.patch('builtins.open', mocker.mock_open())

    tape = Tape('', latency=1)
    assert tape.next().id == 1
    assert tape.current_time == 1
    assert not tape.done
    assert tape.next().id == 2

    # Test basic methods
    assert tape.next().id == 3
    assert tape.current_time == 3
    assert tape.done
    assert tape.next() is None

    # Test reset
    tape = Tape('', latency=1)
    assert tape.next().id == 1
    assert not tape.done
    assert tape.current_time == 1
    assert tape.next().id == 2
    assert tape.next().id == 3
    assert tape.done
    assert tape.current_time == 3

    # Test user orders
    tape = Tape('', latency=1)
    assert tape.next().id == 1
    tape.add_user_order(UserLimitOrder(side='B', price=10000, shares=100))
    assert tape.next().id == 2
    assert tape.next().id == -1
    assert tape.current_time == 2
    assert not tape.done
    assert tape.next().id == 3
    assert tape.done


def test_multiple_user_order(mocker):
    """ Test if there are multiple user orders """
    messages = [
        LimitOrder(1, 1, 'B', 10000, 100),
        LimitOrder(2, 2, 'B', 10000, 100),
        LimitOrder(3, 3, 'B', 10000, 100),
        LimitOrder(4, 4, 'B', 10000, 100),
        LimitOrder(5, 5, 'B', 10000, 100),
    ]

    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=messages)
    mocker.patch('builtins.open', mocker.mock_open())

    tape = Tape('', latency=2)
    assert tape.next().id == 1
    tape.add_user_order(UserLimitOrder(side='B', price=10000, shares=100))
    assert tape.next().id == 2
    tape.add_user_order(UserLimitOrder(side='B', price=10000, shares=100))
    assert tape.next().id == 3
    assert tape.next().id == -1
    assert tape.current_time == 3
    assert tape.next().id == 4
    assert tape.next().id == -2
    assert tape.current_time == 4
    assert tape.next().id == 5
    assert tape.done


def test_sign(mocker):
    """ Test mid price change signs """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=[])
    mocker.patch('builtins.open', mocker.mock_open())

    env = Exchange('', 0, indicators=[])
    book = env.book
    ind = MidPriceDeltaSign(2)
    assert ind.dimension == 2
    book.add_limit_order(LimitOrder(1, 1, 'B', 10000, 100))
    book.add_limit_order(LimitOrder(2, 2, 'S', 12000, 100))
    assert ind.update(env) == (0, 0)

    book.add_limit_order(LimitOrder(3, 3, 'S', 11000, 100))
    assert ind.update(env) == (0, -1)
    assert ind.update(env) == (-1, 0)

    book.match_limit_order(MarketOrder(4, 3, 'B', 100))
    assert ind.update(env) == (0, 1)
    assert ind.update(env) == (1, 0)


def test_position(mocker):
    """ Test position reading """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=[])
    mocker.patch('builtins.open', mocker.mock_open())
    env = Exchange('', 0, indicators=[])
    ind = Position()
    assert ind.dimension == 1
    assert ind.update(env) == (0,)

    env._position = 1000
    assert ind.update(env) == (1000,)


def test_imbalance(mocker):
    """ Test imbalance calculation """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=[])
    mocker.patch('builtins.open', mocker.mock_open())

    env = Exchange('', 0, indicators=[])
    ind = Imbalance(2, decay=0)
    assert ind.dimension == 1
    env.book.add_limit_order(LimitOrder(1, 1, 'B', 10000, 100))
    env.book.add_limit_order(LimitOrder(2, 2, 'S', 12000, 100))
    assert ind.update(env) == (0,)

    env.book.add_limit_order(LimitOrder(3, 3, 'S', 11000, 100))
    assert ind.update(env) == (-100 / 300,)

    env.book.match_limit_order(MarketOrder(4, 3, 'B', 50))
    assert ind.update(env) == (-50 / 250,)

    env.book.add_limit_order(LimitOrder(4, 4, 'B', 9000, 100))
    assert ind.update(env) == (50 / 350,)

    # We only use the first two levels
    env.book.add_limit_order(LimitOrder(5, 5, 'B', 8000, 100))
    assert ind.update(env) == (50 / 350,)

    ind = Imbalance(2, decay=1)
    bid_vol = 100 + 100 * np.exp(1)
    ask_vol = 100 + 50 * np.exp(1)
    assert abs(ind.update(env)[0] - (bid_vol - ask_vol) / (bid_vol + ask_vol)) < 1E-10
