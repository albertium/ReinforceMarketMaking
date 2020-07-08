from dataclasses import dataclass
from datetime import timedelta


def show_time(timestamp):
    return '' if timestamp is None else str(timedelta(microseconds=timestamp / 1000))


@dataclass
class Event:
    """ Base class for order """
    timestamp: int


@dataclass
class LimitOrder(Event):
    """ Limit order """
    id: int
    side: str
    price: int
    shares: int

    def __repr__(self):
        return f'Limit({show_time(self.timestamp)} {self.id} {self.side} {self.price} {self.shares})'


@dataclass
class MarketOrder(Event):
    """ Market order """
    id: int
    side: str
    shares: int

    def __str__(self):
        return f'Market({show_time(self.timestamp)} {self.id} {self.side} {self.shares})'


@dataclass
class CancelOrder(Event):
    """ Cancel order """
    id: int
    shares: int

    def __str__(self):
        return f'Cancel({show_time(self.timestamp)} {self.id} {self.shares})'


@dataclass
class DeleteOrder(Event):
    """ Delete order """
    id: int

    def __str__(self):
        return f'Delete({show_time(self.timestamp)} {self.id})'


@dataclass
class UpdateOrder(Event):
    """ Update order """
    id: int
    old_id: int
    price: int
    shares: int

    def __str__(self):
        return f'Update({show_time(self.timestamp)} {self.old_id}->{self.id} {self.price} {self.shares})'
