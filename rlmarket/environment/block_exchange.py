from __future__ import annotations
from typing import List, Deque, Tuple, Optional, DefaultDict
from math import ceil, floor
from pandas import Timedelta
import numpy as np
from collections import defaultdict

from rlmarket.environment.exchange_elements import Tape, Indicator
from rlmarket.market import OrderBook
from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder
from rlmarket.market import UserLimitOrder, UserMarketOrder, Execution
from rlmarket.environment import Environment, StateT


class BlockExchange(Environment):
    """ Exchange emulate the electronic trading venue """

    tape: Tape

    def __init__(self, files: List[str], indicators: List[Indicator],
                 start_time: int, end_time: int, latency: int = 20_000_000, block_size: int = 50,
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
        self._block_size = block_size + 1

        # Order elements
        self._num_executions = 0
        self._position = 0
        self._position_pnl: int = 0
        self._spread_profit: int = 0
        self._last_spread_profit: int = 0
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
        self.bk_total_pnl: int = 0

    def reset(self) -> StateT:
        """ Reset exchange status """
        print(f'Trading time is from {Timedelta(self._start_time, "ns")} to {Timedelta(self._end_time, "ns")}')
        self._path_pointer = (self._path_pointer + 1) % len(self._paths)
        self.tape = Tape(self._paths[self._path_pointer], latency=self._latency, end_time=self._end_time)

        self._num_executions = 0
        self._position = 0
        self._position_pnl = 0
        self._spread_profit = 0
        self._last_spread_profit = 0
        self.book.reset()

        # Reset training stats
        self.bk_action_counts.clear()
        self.bk_liquidation = 0
        self.bk_bid_counts = 0
        self.bk_ask_counts = 0
        self.bk_spread_profits = []
        self.bk_total_pnl = 0

        # Load market
        while self.tape.current_time < self._start_time:
            self._run_market()

        return self._get_state()

    def step(self, action: int) -> Tuple[StateT, Optional[float], Optional[float], bool]:
        """
        * An block consists of n - 1 episodes
        * The nth episode is used to mark the end of the block.
            * We should remove the spread profit from the last episode because it is purely spread profit but no
                risk from position
        * The risk on leftover position will be counted towards the current block
            * This is because the last execution also marks the first decision point of current block
            * At this decision point, we can choose an asymmetric quotes to reduce position sooner than later.
                Therefore, the risk should be with that the decision  point
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
        if self._wait_for_execution():
            self._num_executions += 1

            # Neutralize position if exceeds limit
            liquidation_ind = 0
            if abs(self._position) >= self._position_limit:
                # Book keeping
                self.bk_liquidation += 1

                # Calculate shares to cover
                shares = int(self._position * self._liquidation_ratio)
                if shares > 0:
                    self.tape.add_user_order(UserMarketOrder(side='S', shares=shares))
                else:
                    self.tape.add_user_order(UserMarketOrder(side='B', shares=abs(shares)))

                if self._wait_for_execution():
                    liquidation_ind = 1
                else:
                    liquidation_ind = -1

            done = liquidation_ind < 0
            if self._num_executions >= self._block_size:
                reward, metric = self._return_reward()
                return self._get_state(), reward, metric, done
            return self._get_state(), None, None, done

        # Book keeping
        return (), None, None, True

    def clean_up(self) -> None:
        # Record last mid price for book keeping
        final_mtm = self.book.mid_price

        while not self.tape.done:
            self._run_market()

        # Check if exchange finishes properly
        if not self.book.empty:
            raise RuntimeError('Market is not fully cleared')

        tmp = {idx: self.bk_action_counts[idx] for idx in range(self.action_space)}
        print(f'Actions: {tmp} | Bids: {self.bk_bid_counts} | Asks: {self.bk_ask_counts} | Cover: {self.bk_liquidation}'
              f' | Avg SP: {np.mean(self.bk_spread_profits) / 10000:.3g}'
              f' | SP: {np.sum(self.bk_spread_profits) / 10000}'
              f' | Pos PnL: {(self.bk_total_pnl + final_mtm * self._position) / 10000}'
              f' | Pos: {self._position}')

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

    def _return_reward(self) -> Tuple[float, float]:
        """ Return new state, reward and metric """
        mark_to_market = self.book.mid_price * self._position
        position_pnl = self._position_pnl + mark_to_market
        dampened_pnl = (position_pnl if position_pnl < 0 else position_pnl * 0.1)
        reward = self._spread_profit - self._last_spread_profit + dampened_pnl
        metric = self._spread_profit + position_pnl

        # Reset cache
        self._num_executions = 0
        self._position_pnl = -mark_to_market
        self._spread_profit = 0
        self._last_spread_profit = 0

        return reward / 10000, metric / 10000

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

    def _wait_for_execution(self) -> bool:
        """ If tape runs out before order is executed, None is returned """
        while not self.tape.done:
            execution = self._run_market()

            if execution:
                # Update position
                mid_price = self.book.mid_price
                self._position += execution.shares
                self._position_pnl -= mid_price * execution.shares
                self._last_spread_profit = (mid_price - execution.price) * execution.shares
                self._spread_profit += self._last_spread_profit

                # Book keeping
                self.bk_total_pnl -= mid_price * execution.shares
                self.bk_spread_profits.append(self._last_spread_profit)
                if execution.shares > 0:
                    self.bk_bid_counts += 1
                else:
                    self.bk_ask_counts += 1

                return True  # Keep going

            if self.tape.current_time > self._end_time:
                # If no execution and end time is passed, return
                return False  # Done

        return False  # Done
