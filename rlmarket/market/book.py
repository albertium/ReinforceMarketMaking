"""
Book is an price-ordered collection of price levels, representing one side of the full order book
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, List, Tuple
from sortedcontainers import SortedList

from rlmarket.market.price_level import PriceLevel
from rlmarket.market.order import LimitOrder, MarketOrder, CancelOrder, DeleteOrder
from rlmarket.market.user_order import UserLimitOrder, UserMarketOrder, Execution


class Book:
    """ Represent bid / ask book """

    def __init__(self, side: str, key_func: Optional[Callable[[int], int]]) -> None:
        self.side = side
        # We need this because price levels follow price priority not time priority (which dict alone can provides)
        self.prices = SortedList(key=key_func)  # Sorted prices
        self.key_func = key_func if key_func else lambda x: x
        self.price_levels: Dict[int, PriceLevel] = {}  # Price to level map
        self.order_pool: Dict[int, PriceLevel] = {}  # Order ID to level map
        self.user_order_pool: Dict[int, PriceLevel] = {}  # Store alive user LimitOrder
        self.front_idx: Optional[int] = None  # Index point to the front price level

    # ========== PriceLevel Operations ==========
    def get_price_level(self, price: int, force_index=False) -> PriceLevel:
        """ Return price level indicated by price. Price level will be added if not already exists """
        level = self.price_levels.get(price, None)
        if level is None:
            self.prices.add(price)
            level = PriceLevel(price)
            self.price_levels[price] = level
            # force_index is used when we are adding a new price level for real order. Order is not added at this point
            #   and shares will be 0. Therefore, we need to force it
            # On the other hand, we still need to run update_front_index for user order because it may change the
            #   ordering
            self.update_front_index(force_index, price)
        return level

    def remove_price_level_if_empty(self, price_level: PriceLevel):
        """ Remove PriceLevel if empty """
        if price_level.empty:
            del self.price_levels[price_level.price]
            # "remove" will raise ValueError if not exists
            self.prices.remove(price_level.price)
            self.update_front_index()

    def update_front_index(self, force_index=False, target_price=None) -> None:
        """ Find out the first price level that has real order """
        for idx, price in enumerate(self.prices):
            if self.price_levels[price].shares > 0 or (force_index and price == target_price):
                self.front_idx = idx
                return
        self.front_idx = None

    def resolve_limit_order_crossing(self, price: int) -> List[Execution]:
        """
        User orders may be placed inside the real market, in which case the newly added real order may cross with the
            user orders. When this happens, we assume that the user orders are executed
        """
        executions = []
        while self.front_price and self.key_func(self.front_price) <= self.key_func(price):
            price_level = self.price_levels[self.front_price]
            for order in price_level.pop_user_orders():
                del self.user_order_pool[order.id]
                executions.append(Execution(order.id, order.price, order.shares))
            self.remove_price_level_if_empty(price_level)
        return executions

    # ========== Order Operations ==========
    def add_limit_order(self, order: LimitOrder) -> None:
        """ Add limit order to the correct price level """
        self.order_pool[order.id] = self.get_price_level(order.price, force_index=True).add_limit_order(order)

    def match_limit_order(self, market_order: MarketOrder) -> Tuple[bool, List[Execution]]:
        """ Match environment order against limit order. Remove empty price level where needed """
        # Sometime environment order may not follow time priority. We should follow the referenced order ID in this case
        user_orders = []
        target_price_level = self.order_pool[market_order.id]

        # User orders may create price levels that do not exist in the real market. Need to match against those first
        while target_price_level.price != self.prices[0]:
            top_level = self.price_levels[self.prices[0]]
            if top_level.num_user_orders != top_level.length:
                raise RuntimeError('Market order being matched against levels not in the front')
            user_orders.extend(self.price_levels[self.prices[0]].pop_user_orders())
            self.remove_price_level_if_empty(top_level)

        # Now get the user orders that are in front of the matched real LimitOrder
        price_level, exhausted, executed_orders = target_price_level.match_limit_order(market_order)
        user_orders.extend(executed_orders)
        self.remove_price_level_if_empty(price_level)

        # Whether the matching limit order is already exhausted
        if exhausted:
            del self.order_pool[market_order.id]

        # Update user order pool
        for user_order in user_orders:
            del self.user_order_pool[user_order.id]

        # Return executions
        executions = [Execution(order.id, order.price, order.shares if order.side == 'B' else -order.shares)
                      for order in user_orders]
        return exhausted, executions

    def cancel_order(self, order: CancelOrder) -> None:
        """ Cancel (partial) shares of a LimitOrder """
        self.order_pool[order.id].cancel_order(order)

    def delete_order(self, order: DeleteOrder):
        """ Delete the whole LimitOrder """
        price_level = self.order_pool[order.id].delete_order(order)
        del self.order_pool[order.id]
        self.remove_price_level_if_empty(price_level)

    # ========== User Order Operation ==========
    def add_user_limit_order(self, order: UserLimitOrder) -> None:
        """ Add user limit order to the correct price level """
        # Right now we only allow one order at a time. Potentially, we can extend to allow multiple orders
        if self.user_order_pool:
            order_id: int
            price_level: PriceLevel
            order_id, price_level = next(iter(self.user_order_pool.items()))

            if order.price != price_level.price:
                price_level.delete_user_order(order_id)
                del self.user_order_pool[order_id]
                self.remove_price_level_if_empty(price_level)
            else:
                # If on the same level, we want to keep the time priority and not update the order
                # TODO: Of course, this is based on the assumption that the order size is the same
                return

        self.user_order_pool[order.id] = self.get_price_level(order.price).add_user_limit_order(order)

    def match_limit_order_for_user(self, order: UserMarketOrder) -> Execution:
        """ Match LimitOrder for UserMarketOrder """
        if self.user_order_pool:
            raise RuntimeError('Cannot execute MarketOrder on the side that also has user LimitOrder')

        total_value = 0
        shares = 0

        # Recall that we are not actually matching the LimitOrders. No need to remove the executed LimitOrder.
        for price in self.prices:
            executed = order.shares - self.price_levels[price].match_limit_order_for_user(order)
            total_value += price * executed
            shares += executed
            if order.shares == 0:
                break

        if order.shares > 0:
            raise RuntimeError('User market order cannot be fully executed')

        return Execution(order.id, int(total_value / shares), shares if order.side == 'B' else -shares)

    # ========= Properties ==========
    # These statistics should not include user orders. Otherwise, we may end up being our own market
    @property
    def quote(self) -> Optional[int]:
        """ Return the front price without user orders """
        if self.front_idx is not None:
            return self.prices[self.front_idx]
        return None

    @property
    def front_price(self) -> Optional[int]:
        """ Front price will include user order price levels """
        if self.prices:
            return self.prices[0]
        return None

    @property
    def volume(self) -> Optional[int]:
        """ Return the volume at the front without user orders """
        if self.front_idx is not None:
            return self.price_levels[self.front_idx].shares
        return None

    def get_depth(self, num_levels: int) -> List[Tuple[int, int]]:
        """ Return the top n price levels without user orders """
        if self.front_idx is not None:
            return [(price, self.price_levels[price].shares)
                    for price in self.prices[self.front_idx: self.front_idx + num_levels]]
        return []
