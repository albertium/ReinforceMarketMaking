from __future__ import annotations
import abc
from typing import List, Deque, Optional, Union, Tuple, TYPE_CHECKING
import pickle
from collections import deque
import numpy as np

from rlmarket.market import Event, UserEvent, OrderBook

if TYPE_CHECKING:
    from rlmarket.environment import Exchange


class Tape:
    """ Provide efficient way to handle real order and user order flow """

    _real_queue: List[Event]
    _num_real_messages: int

    _real_pointer: int
    _user_queue: Deque[UserEvent]
    _curr_time: Optional[int]
    _user_order_id: int

    def __init__(self, path: str, latency: int = 500000, end_time: int = 57570000000000) -> None:
        self._path = path
        self._delay = latency
        self._end_time = end_time
        self._user_queue = deque()

    def reset(self):
        """ Reset tape to original state """
        with open(self._path, 'rb') as f:
            self._real_queue = pickle.load(f)
        self._num_real_messages = len(self._real_queue)

        self._real_pointer = 0
        self._user_queue.clear()
        self._curr_time = None
        self._user_order_id = -1

    def add_user_order(self, order: UserEvent) -> int:
        """ Put user order on tape with correct timestamp and order ID """
        order.timestamp = self._curr_time + self._delay
        order.id = self._user_order_id
        self._user_order_id -= 1
        self._user_queue.append(order)
        return order.id

    def next(self) -> Optional[Union[Event, UserEvent]]:
        """ Return the next order in sorted order """
        if self.done:
            return None

        if self._user_queue:
            ts = self._user_queue[0].timestamp
            if ts < self._real_queue[self._real_pointer].timestamp and ts < self._end_time:
                return self._user_queue.popleft()

        order = self._real_queue[self._real_pointer]
        self._curr_time = order.timestamp
        self._real_pointer += 1
        return order

    @property
    def current_timestamp(self) -> int:
        return self._curr_time

    @property
    def done(self) -> bool:
        return self._real_pointer >= self._num_real_messages


class Indicator(abc.ABC):
    """ Base class for indicator used in exchange to generate states """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    @abc.abstractmethod
    def update(self, env: Exchange) -> Optional[Tuple]:
        """ Update the internal state and return indicator value """

    @property
    def dimension(self):
        return self._dimension


class MidPrice(Indicator):
    """ Return the mid price """

    def __init__(self) -> None:
        super().__init__(dimension=1)

    def update(self, env: Exchange) -> Optional[Tuple]:
        return (env.book.mid_price,)


class MidPriceDeltaSign(Indicator):
    """ Return the signs of mid price changes """

    def __init__(self, lags: int) -> None:
        super().__init__(dimension=lags)
        self.lags = lags
        self.store: Deque[int] = deque([0] * lags)
        self.last_price: Optional[int] = None

    def update(self, env: Exchange) -> Optional[Tuple]:
        book = env.book

        if self.last_price is None:
            self.last_price = book.mid_price

        mid_price = book.mid_price
        self.store.append(np.sign(mid_price - self.last_price))
        self.last_price = mid_price

        if len(self.store) > self.lags:
            self.store.popleft()

        if len(self.store) == self.lags:
            return tuple(self.store)

        return None


class HalfSpread(Indicator):

    def __init__(self) -> None:
        super().__init__(dimension=1)

    def update(self, env: Exchange) -> Optional[Tuple]:
        return (int(env.book.spread / 2),)


class Position(Indicator):
    """ Return the current accumulative position """

    def __init__(self) -> None:
        super().__init__(dimension=1)

    def update(self, env: Exchange) -> Optional[Tuple]:
        return (env.position,)


class Imbalance(Indicator):
    """ Return the bia-ask imbalance """

    def __init__(self, num_levels: int = 3, decay: float = 0.5) -> None:
        super().__init__(dimension=1)
        self.num_levels = num_levels
        self.weights = [np.exp(-decay * idx) for idx in range(num_levels)]

    def update(self, env: Exchange) -> Optional[Tuple]:
        bid_depths, ask_depths = env.book.get_depth(self.num_levels)
        bid_volume = sum(level[1] * weight for level, weight in zip(bid_depths, self.weights))
        ask_volume = sum(level[1] * weight for level, weight in zip(ask_depths, self.weights))
        return (bid_volume - ask_volume) / (bid_volume + ask_volume),


class RemainingTime(Indicator):
    """ Remaining time in relation to end time in range of [0, 1] """

    def __init__(self, start_time: int, end_time: int) -> None:
        super().__init__(dimension=1)
        self.normalization = end_time - start_time
        self.end_time = end_time

    def update(self, env: Exchange) -> Optional[Tuple]:
        return ((self.end_time - env.tape.current_timestamp) / self.normalization,)
