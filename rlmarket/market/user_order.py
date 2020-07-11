from dataclasses import dataclass


@dataclass
class UserEvent:
    """ Base class for user order """
    timestamp: int = None
    id: int = None


@dataclass
class UserLimitOrder(UserEvent):
    """ LimitOrder originated from user """
    side: str = None
    price: int = None
    shares: int = None


@dataclass
class UserMarketOrder(UserEvent):
    """ MarketOrder originated from user """
    side: str = None
    shares: int = None


@dataclass
class Execution:
    id: int
    price: int
    shares: int  # Negative shares means short / sell
