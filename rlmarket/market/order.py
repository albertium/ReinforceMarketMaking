"""
Real orders
"""
from dataclasses import dataclass
from datetime import timedelta


def show_time(timestamp):
    return '' if timestamp is None else str(timedelta(microseconds=timestamp / 1000))


@dataclass
class Event:
    """ Base class for real order """
    timestamp: int
    id: int


@dataclass
class LimitOrder(Event):
    """ Limit order """
    side: str
    price: int
    shares: int

    def __repr__(self):
        return f'Limit({show_time(self.timestamp)} {self.id} {self.side} {self.price} {self.shares})'


@dataclass
class MarketOrder(Event):
    """ Market order """
    side: str
    shares: int

    def __str__(self):
        return f'Market({show_time(self.timestamp)} {self.id} {self.side} {self.shares})'


@dataclass
class CancelOrder(Event):
    """ Cancel order """
    shares: int

    def __str__(self):
        return f'Cancel({show_time(self.timestamp)} {self.id} {self.shares})'


@dataclass
class DeleteOrder(Event):
    """ Delete order """

    def __str__(self):
        return f'Delete({show_time(self.timestamp)} {self.id})'


@dataclass
class UpdateOrder(Event):
    """ Update order """
    old_id: int
    price: int
    shares: int

    def __str__(self):
        return f'Update({show_time(self.timestamp)} {self.old_id}->{self.id} {self.price} {self.shares})'
