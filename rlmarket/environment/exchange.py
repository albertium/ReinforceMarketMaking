from __future__ import annotations
from typing import List, Union, Deque, Tuple, Optional
from math import ceil, floor
from pandas import Timedelta

from rlmarket.environment.exchange_elements import Tape, Indicator
from rlmarket.market import OrderBook
from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder
from rlmarket.market import UserLimitOrder, UserMarketOrder, Execution
from rlmarket.environment import Environment, StateT


class Exchange(Environment):
    """ Exchange emulate the electronic trading venue """

    def __init__(self, ticker: str, start_time: int, indicators: List[Indicator],
                 num_episodes: int = 100, gamma: float = 0.99,
                 order_size: int = 100, position_limit: int = 10000, liquidation_ratio: float = 0.2,
                 latency: int = 20_000_000) -> None:
        super().__init__(is_block_training=True)

        # Load real message data

        self.tape = Tape(f'C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/data/parsed/{ticker}_20170201.pickle',
                         latency=latency)
        self.book = OrderBook()
        self.start_time = start_time
        print(f'Start time is {Timedelta(start_time, "ns")}')
        self.indicators = indicators

        # User properties
        self.position = 0
        self.order_size = order_size
        self.position_limit = position_limit

        self.num_episodes = num_episodes  # Number of sub episodes to make up a full episode
        self.decays = list(reversed([gamma ** i for i in range(num_episodes)]))
        self.liquidation_ratio = liquidation_ratio  # The portion of position to neutralize when using MarketOrder

        self.total_value: float = 0  # Total reward of a block
        self.spread_profit: float = 0  # Profit from limit order price in relation to mid price
        self.episode_count: int = 0  # Count the number of episodes in the current block

    def reset(self) -> StateT:
        """ Reset exchange status """
        self.position = 0
        self.book.reset()
        self.tape.reset()
        self._preload_market()
        return self._get_state()

    def step(self, action: int) -> Tuple[StateT, Optional[Union[float, List[float]]], bool]:
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

        # Wait for result
        if self._wait_for_execution():
            self.episode_count += 1

            # Neutralize position if exceeds limit
            if abs(self.position) >= self.position_limit:
                shares = int(self.position * self.liquidation_ratio)
                if shares > 0:
                    self.tape.add_user_order(UserMarketOrder(side='S', shares=shares))
                else:
                    self.tape.add_user_order(UserMarketOrder(side='B', shares=abs(shares)))

                self._wait_for_execution()  # Liquidate position

            if self.episode_count >= self.num_episodes:
                # Calculate rewards
                final_value = self.total_value + self.book.mid_price * self.position
                unrealized = final_value - self.spread_profit
                # Asymmetric reward on position
                reward = max(unrealized, 0) * 0.1 + min(unrealized, 0) + self.spread_profit
                rewards = [reward * decay for decay in self.decays]
                rewards.append(rewards[-1] / 10000)

                # Reset stats
                self.total_value = 0
                self.spread_profit = 0
                self.episode_count = 0

                return self._get_state(), rewards, self.tape.done

        # If tape is done or not enough episodes are gathered
        if self.tape.done:
            if not self.book.empty:
                raise RuntimeError('Market is not fully cleared')
            return (), None, True
        return self._get_state(), None, False

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        """ To do later """

    @property
    def action_space(self) -> int:
        return 9

    @property
    def state_dimension(self) -> int:
        return 3

    # ========== Private Methods ==========
    def _get_state(self) -> StateT:
        """ Define the state of exchange """
        return sum((ind.update(self) for ind in self.indicators), ())

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
        self.tape.add_user_order(UserLimitOrder(side='B', price=bid_price, shares=self.order_size))
        self.tape.add_user_order(UserLimitOrder(side='S', price=ask_price, shares=self.order_size))

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

    def _preload_market(self):
        """ Run the market until the trading start time """
        while True:
            self._run_market()
            if self.tape.current_timestamp >= self.start_time:
                break

    def _wait_for_execution(self) -> bool:
        """ If tape runs out before order is executed, None is returned """
        while not self.tape.done:
            execution = self._run_market()

            if execution:
                self.total_value += -execution.price * execution.shares  # Cash is opposite of position
                self.spread_profit += (self.book.mid_price - execution.price) * execution.shares
                self.position += execution.shares
                return True  # Wait successfully

        return False
