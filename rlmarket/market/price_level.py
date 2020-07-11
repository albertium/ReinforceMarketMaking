"""
Price level represent the queue at a certain price level in the bid / ask book
"""
from __future__ import annotations
from typing import Dict, Tuple, Union, List, Set

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
        self.user_orders: Set[int] = set()

    # ========== Order Operations ===========
    def add_limit_order(self, order: LimitOrder) -> PriceLevel:
        """ Add limit order to the queue """
        if order.price != self.price:
            raise RuntimeError(f'LimitOrder price {order.price} is not the same as PriceLevel price {self.price}')
        self.shares += order.shares
        self.queue[order.id] = order
        return self

    def match_limit_order(self, market_order: MarketOrder) -> Tuple[PriceLevel, bool, List[UserLimitOrder]]:
        """ Match against a limit order in the queue """

        # Order ID of MarketOrder is the LimitOrder ID to be matched
        limit_order = self.queue[market_order.id]

        if market_order.side == limit_order.side:
            raise RuntimeError(f'LimitOrder and MarketOrder are on the same side ({market_order.side})')

        # Handle user order
        user_orders = []
        for order in self.queue.values():
            if isinstance(order, UserLimitOrder):
                user_orders.append(order)
            else:
                # If the real LimitOrder is not at the front, then this real order is just an abnormal execution.
                #   We should not match against user LimitOrder in this case
                if order != limit_order:
                    user_orders.clear()
                break

        # Remove user orders if any
        for order in user_orders:
            del self.queue[order.id]
            self.user_orders.remove(order.id)

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
        return self, exhausted, user_orders

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

        # No need to increase PriceLevel shares since user orders are phantom orders
        self.queue[order.id] = order
        self.user_orders.add(order.id)
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
        if self.user_orders:
            raise RuntimeError('User market order cannot match against price level that has user limit order')
        order.shares = max(0, order.shares - self.shares)
        return order.shares  # Return remaining shares

    def pop_user_orders(self) -> List[UserLimitOrder]:
        """ Clear out and return all user orders """
        orders = []
        for order_id in self.user_orders:
            orders.append(self.queue[order_id])
            del self.queue[order_id]
        self.user_orders.clear()
        return orders

    def delete_user_order(self, order_id: int) -> None:
        """
        Real DeleteOrder is used to indicate action intention.
        However, we know what the intention is with user orders. Therefore, we do not need DeleteOrder
        """
        del self.queue[order_id]
        self.user_orders.remove(order_id)

    # ========== Properties ==========
    @property
    def length(self):
        """ Length (total number) of orders (real and user) """
        return len(self.queue)

    @property
    def num_user_orders(self):
        """ Number of user orders """
        return len(self.user_orders)

    @property
    def empty(self):
        """ Whether PriceLevel is empty """
        return self.shares == 0 and not self.user_orders
