"""
Price level represent the queue at a certain price level in the bid / ask book
"""
from __future__ import annotations
from typing import Dict, Tuple, Union, Optional

from rlmarket.market.order import LimitOrder, MarketOrder, CancelOrder, DeleteOrder
from rlmarket.market.user_order import UserLimitOrder, UserMarketOrder


class PriceLevel:
    """ Price level is the queue of a price level """

    def __init__(self, price: int) -> None:
        """ We won't track user orders on PriceLevel since it is a book level enforcement """
        self.price = price
        self.shares = 0
        # Rely on the feature of dict that it preserves insertion order
        self.queue: Dict[int, Union[LimitOrder, UserLimitOrder]] = {}
        self.user_order_id: Optional[int] = None

    # ========== Order Operations ===========
    def add_limit_order(self, order: LimitOrder) -> PriceLevel:
        """ Add limit order to the queue """
        if order.price != self.price:
            raise RuntimeError(f'LimitOrder price {order.price} is not the same as PriceLevel price {self.price}')
        self.shares += order.shares
        self.queue[order.id] = order
        return self

    def match_limit_order(self, market_order: MarketOrder) -> Tuple[PriceLevel, bool, Optional[UserLimitOrder]]:
        """ Match against a limit order in the queue """

        # Order ID of MarketOrder is the LimitOrder ID to be matched
        limit_order = self.queue[market_order.id]

        if market_order.side == limit_order.side:
            raise RuntimeError(f'LimitOrder and MarketOrder are on the same side ({market_order.side})')

        # Handle user order
        user_order = None
        if self.user_order_id is not None:
            iterator = iter(self.queue)
            if next(iterator) == self.user_order_id and next(iterator) == market_order.id:
                user_order = self.pop_user_order()

        # Handle real order
        if limit_order.shares > market_order.shares:
            self.shares -= market_order.shares
            limit_order.shares -= market_order.shares
            exhausted = False

        elif limit_order.shares == market_order.shares:
            self.shares -= market_order.shares
            del self.queue[market_order.id]
            exhausted = True

        else:
            raise RuntimeError(f'Market order shares {market_order.shares} is more than '
                               f'limit order shares {limit_order.shares}')

        # Return self for convenience of book level operation
        return self, exhausted, user_order

    def cancel_order(self, order: CancelOrder) -> None:
        """ Process order cancellation """
        if self.queue[order.id].shares <= order.shares:
            raise RuntimeError('Cancel more shares than available')
        self.queue[order.id].shares -= order.shares
        self.shares -= order.shares

    def delete_order(self, order: DeleteOrder) -> PriceLevel:
        """" Process order deletion """
        self.shares -= self.queue[order.id].shares
        del self.queue[order.id]
        return self

    # ========== User Order Operation ==========
    def add_user_limit_order(self, order: UserLimitOrder) -> PriceLevel:
        """ Add user LimitOrder to the queue """
        if order.id >= 0:
            raise ValueError('User order should have negative order ID')

        if order.price != self.price:
            raise ValueError(f'User LimitOrder price {order.price} is not the same as PriceLevel {self.price}')

        if self.user_order_id is not None:
            raise RuntimeError(f'Another user LimitOrder already exists')

        # No need to increase PriceLevel shares since user orders are phantom orders
        self.queue[order.id] = order
        self.user_order_id = order.id
        return self

    def match_limit_order_for_user(self, order: UserMarketOrder) -> int:
        """
        Match user MarketOrder against LimitOrders.
        * User cannot put MarketMarket against the side that it already has LimitOrder in.
        * This check will be enforced at the Book level. Therefore, at PriceLevel level, we can assume user MarketOrder
            never match against user LimitOrder.
        * The check is on Book level because user cannot put MarketOrder against the whole side, not just individual
            PriceLevel that has user LimitOrder
        """
        if self.user_order_id is not None:
            raise RuntimeError('User market order cannot match against price level that has user limit order')
        order.shares = max(0, order.shares - self.shares)
        return order.shares  # Return remaining shares

    def pop_user_order(self) -> UserLimitOrder:
        """ Remove and return user order """
        order = self.queue.pop(self.user_order_id)
        self.user_order_id = None
        return order

    # ========== Properties ==========
    @property
    def length(self):
        """ Length (total number) of orders (real and user) """
        return len(self.queue)

    @property
    def empty(self):
        """ Whether PriceLevel is empty """
        return self.shares == 0 and self.user_order_id is None
