from __future__ import annotations
from typing import List, Union, Deque, Tuple, Optional, DefaultDict
from math import ceil, floor
from pandas import Timedelta
import numpy as np
from collections import defaultdict

from rlmarket.environment.exchange_elements import Tape, Indicator
from rlmarket.market import OrderBook
from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder
from rlmarket.market import UserLimitOrder, UserMarketOrder, Execution
from rlmarket.environment import Environment, StateT


class Exchange(Environment):
    """ Exchange emulate the electronic trading venue """

    tape: Tape

    def __init__(self, files: List[str], indicators: List[Indicator],
                 start_time: int, end_time: int, latency: int = 20_000_000,
                 order_size: int = 100, position_limit: int = 10000, liquidation_ratio: float = 0.2) -> None:

        # Data elements
        self._paths = [f'C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/data/parsed/{file}.pickle'
                       for file in files]

        self._indicators = indicators
        self._path_pointer = -1  # Point to the file to use

        # Time elements
        self._start_time = start_time
        self._end_time = end_time
        self._latency = latency

        # Order elements
        self._last_position_pnl: int = 0
        self._last_spread_profit: int = 0
        self._position = 0
        self._order_size = order_size
        self._position_limit = position_limit
        self._liquidation_ratio = liquidation_ratio  # The portion of position to neutralize when using MarketOrder

        # Set up market
        self.book = OrderBook()

        # Book keeping stats
        self.bk_action_counts: DefaultDict[int, int] = defaultdict(int)
        self.bk_liquidation: int = 0
        self.bk_bid_counts: int = 0
        self.bk_ask_counts: int = 0
        self.bk_spread_profits: List[int] = []

    def reset(self) -> StateT:
        """ Reset exchange status """
        print(f'Trading time is from {Timedelta(self._start_time, "ns")} to {Timedelta(self._end_time, "ns")}')
        self._path_pointer = (self._path_pointer + 1) % len(self._paths)
        self.tape = Tape(self._paths[self._path_pointer], latency=self._latency, end_time=self._end_time)

        self._position = 0
        self._last_position_pnl = 0
        self._last_spread_profit = 0
        self.book.reset()

        # Reset training stats
        self.bk_action_counts.clear()
        self.bk_liquidation = 0
        self.bk_bid_counts = 0
        self.bk_ask_counts = 0
        self.bk_spread_profits = []

        # Load market
        while self.tape.current_time < self._start_time:
            self._run_market()

        return self._get_state()

    def step(self, action: int) -> Tuple[StateT, float, float, bool]:
        """
        * An episode is from placing order pair to one of the orders being matched
        * New order will replaced the old order where needed without sacrificing time priority (by using alias)
        * Only one order is allowed at a time
        * Neutralizing is taken automatically at the end of step whenever position is on or beyond limit
        * Update agent every n steps to let the position effect to manifest. We should calculate the total reward
            after n steps and assigns that back to all action in the block with a discount factor
        * The reason we do not want to separate spread profit from position unrealized PnL is that we will
            over-emphasize the spread profit because position PnL is discounted while spread profit is not in this case.
        """
        # Perform action
        if action < 5:
            # Symmetric market making
            self._place_order(action + 1, action + 1)
        elif action == 5:
            self._place_order(3, 1)
        elif action == 6:
            self._place_order(1, 3)
        elif action == 7:
            self._place_order(5, 2)
        elif action == 8:
            self._place_order(2, 5)
        else:
            raise RuntimeError(f'Unrecognized action {action}')

        self.bk_action_counts[action] += 1

        # Wait for result
        reward_pair = self._wait_for_execution()
        if reward_pair is not None:

            # Neutralize position if exceeds limit
            if abs(self._position) >= self._position_limit:
                # Book keeping
                self.bk_liquidation += 1

                # Calculate shares to cover
                shares = int(self._position * self._liquidation_ratio)
                if shares > 0:
                    self.tape.add_user_order(UserMarketOrder(side='S', shares=shares))
                else:
                    self.tape.add_user_order(UserMarketOrder(side='B', shares=abs(shares)))

                liquidation_pair = self._wait_for_execution()  # Liquidate position
                if liquidation_pair is not None:
                    position_pnl, spread_profit = liquidation_pair[1] - liquidation_pair[2], liquidation_pair[2]
                    self._last_position_pnl += position_pnl
                    self._last_spread_profit += spread_profit

                    return self._get_state(), reward_pair[0] / 10000, reward_pair[1] / 10000, False
                return (), reward_pair[0] / 10000, reward_pair[1] / 10000, True

            return self._get_state(), reward_pair[0] / 10000, reward_pair[1] / 10000, False
        return (), 0, 0, True

    def clean_up(self) -> None:
        while not self.tape.done:
            self._run_market()

        # Check if exchange finishes properly
        if not self.book.empty:
            raise RuntimeError('Market is not fully cleared')

        tmp = {idx: self.bk_action_counts[idx] for idx in range(9)}
        print(f'Actions: {tmp} | Bids: {self.bk_bid_counts} | Asks: {self.bk_ask_counts} '
              f'| Cover: {self.bk_liquidation} | Avg Profit: {np.mean(self.bk_spread_profits) / 10000}')

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        """ To do later """

    @property
    def action_space(self) -> int:
        return 9

    @property
    def state_dimension(self) -> int:
        return sum([ind.dimension for ind in self._indicators], 0)

    @property
    def position(self) -> int:
        return self._position

    # ========== Helpers ==========
    def _get_state(self) -> StateT:
        """ Define the state of exchange """
        return sum((ind.update(self) for ind in self._indicators), ())

    def _place_order(self, bid_dist: int, ask_dist: int) -> None:
        """
        * Basically, we only need a fancy UpdateOrder which place a LimitOrder at the specified price and cancel the
            original order if it exists when the UpdateOrder hit the market.
        * Sometime, the remaining order may be executed after the next action is issued and before the fancy LimitOrder
            hit the market, due to latency. This is fine. We just attribute the profit to the action that originates
            it.
        """
        # We try not to place order inside the market
        spread = ceil(self.book.spread / 2)
        mid_price = self.book.mid_price
        # Round to one cent
        bid_price = floor((mid_price - bid_dist * spread) / 100) * 100
        ask_price = ceil((mid_price + ask_dist * spread) / 100) * 100

        # "Fancy" LimitOrder will delete the existing one if it does exist when the new LimitOrder hits the market
        self.tape.add_user_order(UserLimitOrder(side='B', price=bid_price, shares=self._order_size))
        self.tape.add_user_order(UserLimitOrder(side='S', price=ask_price, shares=self._order_size))

    def _run_market(self) -> Optional[Execution]:
        """ Update OrderBook for an event """
        order = self.tape.next()

        if isinstance(order, LimitOrder):
            return self.book.add_limit_order(order)
        elif isinstance(order, MarketOrder):
            return self.book.match_limit_order(order)
        elif isinstance(order, CancelOrder):
            return self.book.cancel_order(order)
        elif isinstance(order, DeleteOrder):
            return self.book.delete_order(order)
        elif isinstance(order, UpdateOrder):
            return self.book.modify_order(order)
        elif isinstance(order, UserLimitOrder):
            return self.book.add_user_limit_order(order)
        elif isinstance(order, UserMarketOrder):
            return self.book.match_limit_order_for_user(order)
        else:
            raise ValueError(f'Unrecognized order type {type(order)}')

    def _wait_for_execution(self) -> Optional[Tuple[float, float, float]]:
        """ If tape runs out before order is executed, None is returned """
        while not self.tape.done:
            execution = self._run_market()

            if execution:
                # Calculate profit for last episode
                mid_price = self.book.mid_price
                # Calculate PnL before position update
                position_pnl = mid_price * self._position + self._last_position_pnl
                scaled_pnl = position_pnl * 0.1 if position_pnl > 0 else position_pnl
                last_spread_profit = self._last_spread_profit
                last_reward = (last_spread_profit + scaled_pnl)
                last_profit = (last_spread_profit + position_pnl)

                # Update for current episode
                self._position += execution.shares
                # Calculate pnl after position update
                self._last_position_pnl = -mid_price * self._position
                self._last_spread_profit = (self.book.mid_price - execution.price) * execution.shares

                # Book keeping
                self.bk_spread_profits.append(self._last_spread_profit)
                if execution.shares > 0:
                    self.bk_bid_counts += 1
                else:
                    self.bk_ask_counts += 1

                return last_reward, last_profit, last_spread_profit

            if self.tape.current_time > self._end_time:
                # If no execution and end time is passed, return
                return None

        return None
