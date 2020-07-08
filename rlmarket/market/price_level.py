"""
Price level represent the queue at a certain price level in the bid / ask book
"""
from __future__ import annotations
from typing import Dict, Tuple

from rlmarket.market.order import LimitOrder, MarketOrder, CancelOrder, DeleteOrder


class PriceLevel:
    """ Price level is the queue of a price level """

    def __init__(self, price: int) -> None:
        self.price = price
        # TODO: This is not currently used. To be removed
        self.num_orders = 0
        self.shares = 0
        # Rely on the feature of dict that it preserves insertion order
        self.queue: Dict[int, LimitOrder] = {}

    def add_limit_order(self, order: LimitOrder) -> PriceLevel:
        """ Add limit order to the queue """
        if order.price != self.price:
            raise RuntimeError(f'LimitOrder price {order.price} is not the same as PriceLevel price {self.price}')
        self.num_orders += 1
        self.shares += order.shares
        self.queue[order.id] = order
        return self

    def match_limit_order(self, order: MarketOrder) -> Tuple[PriceLevel, bool]:
        """ Match against a limit order in the queue """

        # Order ID of MarketOrder is the LimitOrder ID to be matched
        limit_order = self.queue[order.id]

        if order.side == limit_order.side:
            raise RuntimeError(f'LimitOrder and MarketOrder are on the same side ({order.side})')

        if limit_order.shares > order.shares:
            self.shares -= order.shares
            limit_order.shares -= order.shares
            exhausted = False

        elif limit_order.shares == order.shares:
            self.num_orders -= 1
            self.shares -= order.shares
            del self.queue[order.id]
            exhausted = True

        else:
            raise RuntimeError(f'Market order shares {order.shares} is more than '
                               f'limit order shares {limit_order.shares}')

        # Return self for convenience of book level operation
        return self, exhausted

    def cancel_order(self, order: CancelOrder) -> None:
        """ Process order cancellation """
        if self.queue[order.id].shares <= order.shares:
            raise RuntimeError('Cancel more shares than available')
        self.queue[order.id].shares -= order.shares
        self.shares -= order.shares

    def delete_order(self, order: DeleteOrder) -> PriceLevel:
        """" Process order deletion """
        self.num_orders -= 1
        self.shares -= self.queue[order.id].shares
        del self.queue[order.id]
        return self
