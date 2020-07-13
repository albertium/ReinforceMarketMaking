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
        self.key_func = key_func if key_func else lambda x: x

        # We need this because price levels follow price priority not time priority (which dict alone can provides)
        self.prices = SortedList(key=key_func)  # Sorted prices
        self.price_levels: Dict[int, PriceLevel] = {}  # Price to level map
        self.order_pool: Dict[int, PriceLevel] = {}  # Order ID to level map
        # Store original order ID, new order alias and PriceLevel
        # TODO: We rely on the fact there is only one user order.
        self.user_order_pool: Optional[Tuple[int, int, PriceLevel]] = None
        self._front_idx: Optional[int] = None  # Index point to the front price level

    def reset(self):
        self.prices.clear()
        self.price_levels.clear()
        self.order_pool.clear()
        self.user_order_pool = None
        self._front_idx = None

    # ========== Order Operations ==========
    def add_limit_order(self, order: LimitOrder) -> None:
        """ Add limit order to the correct price level """
        if order.id in self.order_pool:
            raise RuntimeError(f'LimitOrder {order.id} already exists')

        self.order_pool[order.id] = self._get_price_level(order.price, force_index=True).add_limit_order(order)

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
            self._remove_price_level_if_empty(top_level)

        # Now get the user orders that are in front of the matched real LimitOrder
        price_level, exhausted, executed_orders = target_price_level.match_limit_order(market_order)
        user_orders.extend(executed_orders)
        self._remove_price_level_if_empty(price_level)

        # Whether the matching limit order is already exhausted
        if exhausted:
            del self.order_pool[market_order.id]

        # Update user order pool and return executions
        return exhausted, [self._handle_matched_user_limit_order(order) for order in user_orders]

    def cancel_order(self, order: CancelOrder) -> None:
        """ Cancel (partial) shares of a LimitOrder """
        self.order_pool[order.id].cancel_order(order)

    def delete_order(self, order: DeleteOrder):
        """ Delete the whole LimitOrder """
        price_level = self.order_pool[order.id].delete_order(order)
        del self.order_pool[order.id]
        self._remove_price_level_if_empty(price_level)

    # ========== User Order Operation ==========
    def add_user_limit_order(self, order: UserLimitOrder) -> Optional[int]:
        """ Add user limit order to the correct price level """
        # TODO: currently we only allow one order at a time. Potentially, we can extend to allow multiple orders
        if self.user_order_pool:
            original_id, alias, price_level = self.user_order_pool

            if order.price != price_level.price:
                # Prices are different. Should replace the old order completely
                price_level.delete_user_order(original_id)
                self._remove_price_level_if_empty(price_level)
                new_price_level = self._get_price_level(order.price).add_user_limit_order(order)
                self.user_order_pool = order.id, order.id, new_price_level
            else:
                # If on the same level, we want to keep the time priority and not actually update the order
                # TODO: Of course, this is based on the assumption that the order size is the same
                self.user_order_pool = original_id, order.id, price_level

            return alias  # Signal the upstream process that the previous order is removed

        else:
            price_level = self._get_price_level(order.price).add_user_limit_order(order)
            self.user_order_pool = order.id, order.id, price_level
            return None

    def resolve_book_crossing_on_user_orders(self, price: int) -> List[Execution]:
        """
        User orders may be placed inside the real market, in which case the newly added real order may cross with the
            user orders. When this happens, we assume that the user orders are executed
        """
        signed_price = self.key_func(price)

        if self.quote and self.key_func(self.quote) <= signed_price:
            raise RuntimeError('Real order crosses real order')

        executions = []
        while self.front_price and self.key_func(self.front_price) <= signed_price:
            price_level = self.price_levels[self.front_price]
            executions.extend(self._handle_matched_user_limit_order(order) for order in price_level.pop_user_orders())
            self._remove_price_level_if_empty(price_level)
        return executions

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

    # ========== Private Methods ==========
    def _get_price_level(self, price: int, force_index=False) -> PriceLevel:
        """ Return price level indicated by price. Price level will be added if not already exists """
        level = self.price_levels.get(price, None)

        # shares == 0 means that the PriceLevel was previously occupied by user order only
        if level is None:
            self.prices.add(price)
            level = PriceLevel(price)
            self.price_levels[price] = level
            # force_index is used when we are adding a new price level for real order. Order is not added at this point
            #   and shares will be 0. Therefore, we need to force it
            # On the other hand, we still need to run update_front_index for user order because it may change the
            #   ordering
            self._update_front_index(force_index, price)

        elif level.shares == 0:
            self._update_front_index(force_index, price)

        return level

    def _remove_price_level_if_empty(self, price_level: PriceLevel):
        """ Remove PriceLevel if empty """
        if price_level.empty:
            del self.price_levels[price_level.price]
            # "remove" will raise ValueError if not exists
            self.prices.remove(price_level.price)

        if price_level.shares == 0:
            # Separate from the logic above because we run be in the situation where real orders are exhausted
            #   but at least one user order is waiting. In this case, this price level is technically gone
            self._update_front_index()

    def _update_front_index(self, force_index=False, target_price=None) -> None:
        """ Find out the first price level that has real order """
        for idx, price in enumerate(self.prices):
            if self.price_levels[price].shares > 0 or (force_index and price == target_price):
                self._front_idx = idx
                return
        self._front_idx = None

    def _handle_matched_user_limit_order(self, order: UserLimitOrder) -> Execution:
        """ Book-keeping actions for UserLimitOrder execution """
        # We may have renamed the order ID already. Instead of replacing the original order directly and losing
        #   time priority, we do a post-processing to map to the original order to the new ID
        alias = self.user_order_pool[1]
        self.user_order_pool = None
        return Execution(alias, order.price, order.shares if self.side == 'B' else -order.shares)

    # ========== Properties ==========
    # These statistics should not include user orders. Otherwise, we may end up being our own market
    @property
    def quote(self) -> Optional[int]:
        """ Return the front price without user orders """
        if self._front_idx is not None:
            return self.prices[self._front_idx]
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
        if self._front_idx is not None:
            return self.price_levels[self.quote].shares
        return None

    def get_depth(self, num_levels: int) -> List[Tuple[int, int]]:
        """ Return the top n price levels without user orders """
        if self._front_idx is not None:
            return [(price, self.price_levels[price].shares)
                    for price in self.prices[self._front_idx: self._front_idx + num_levels]]
        return []
