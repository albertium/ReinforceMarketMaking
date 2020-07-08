"""
Book is an price-ordered collection of price levels, representing one side of the full order book
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, List, Tuple
from sortedcontainers import SortedList

from rlmarket.market.price_level import PriceLevel
from rlmarket.market.order import LimitOrder, MarketOrder, CancelOrder, DeleteOrder


class Book:
    """ Represent bid / ask book """

    def __init__(self, side: str, key_func: Optional[Callable[[int], int]]) -> None:
        self.side = side
        self.prices = SortedList(key=key_func)  # Sorted prices
        self.price_levels: Dict[int, PriceLevel] = {}  # Price to level map
        self.order_pool: Dict[int, PriceLevel] = {}  # Order ID to level map

    # ========== PriceLevel Operations ==========
    def get_price_level(self, price: int) -> PriceLevel:
        """ Return price level indicated by price. Price level will be added if not already exists """
        level = self.price_levels.get(price, None)
        if level is None:
            self.prices.add(price)
            level = PriceLevel(price)
            self.price_levels[price] = level
        return level

    def remove_price_level(self, level: PriceLevel):
        del self.price_levels[level.price]
        # "remove" will raise ValueError if not exists
        self.prices.remove(level.price)

    # ========== Order Operations ==========
    def add_limit_order(self, order: LimitOrder) -> None:
        """ Add limit order to the correct price level """
        self.order_pool[order.id] = self.get_price_level(order.price).add_limit_order(order)

    def match_limit_order(self, market_order: MarketOrder) -> bool:
        """ Match environment order against limit order. Remove empty price level where needed """
        # Sometime environment order may not follow time priority. We should follow the referenced order ID in this case
        price_level, exhausted = self.order_pool[market_order.id].match_limit_order(market_order)

        if price_level.shares == 0:
            self.remove_price_level(price_level)

        # Whether the matching limit order is already exhausted
        if exhausted:
            del self.order_pool[market_order.id]

        return exhausted

    def cancel_order(self, order: CancelOrder) -> None:
        """ Cancel (partial) shares of a LimitOrder """
        self.order_pool[order.id].cancel_order(order)

    def delete_order(self, order: DeleteOrder):
        """ Delete the whole LimitOrder """
        price_level = self.order_pool[order.id].delete_order(order)
        del self.order_pool[order.id]
        if price_level.shares == 0:
            self.remove_price_level(price_level)

    # ========= Properties ==========
    @property
    def quote(self) -> Optional[int]:
        """ Return the front price """
        if self.prices:
            return self.prices[0]
        return None

    def get_depth(self, num_levels: int) -> List[Tuple[int, int]]:
        """ Return the top n price levels """
        return [(price, self.price_levels[price].shares) for price in self.prices[:num_levels]]
