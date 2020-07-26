"""
Exchange environment based on gym env
"""
from typing import List, Deque, DefaultDict, Optional, Tuple
from collections import deque, defaultdict
from gym import Env, spaces
import numpy as np
from pandas import Timedelta

from rlmarket.environment.exchange_elements import Tape, Indicator
from rlmarket.market import OrderBook
from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder
from rlmarket.market import UserLimitOrder, UserMarketOrder, Execution


class Exchange(Env):
    """ Exchange environment that implements the gym env API """

    tape: Tape

    def __init__(self, files: List[str], indicators: List[Indicator],
                 reward_lb: float, reward_ub: float,
                 start_time: int, end_time: int, latency: int = 20_000_000,
                 order_size: int = 100, position_limit: int = 10000, liquidation_ratio: float = 0.2) -> None:

        if reward_lb >= 0:
            raise ValueError(f'Reward lower bound {reward_lb} should be negative')

        # Specify gym env specs
        self.action_space = spaces.Discrete(4)
        state_dimension = sum([ind.dimension for ind in indicators], 0)
        self.observation_space = spaces.Box(low=-3, high=3, shape=(state_dimension,), dtype=np.float32)

        # Data elements
        self._paths = [f'C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/data/parsed/{file}.pickle'
                       for file in files]

        self._indicators = indicators
        self._path_pointer = -1  # Point to the file to use

        # Reward parameters
        self._reward_lb = reward_lb * order_size
        self._reward_ub = reward_ub * order_size

        # Time elements
        self._start_time = start_time
        self._end_time = end_time
        self._latency = latency

        # Order elements
        self._open_positions: Deque[Execution] = deque()
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
        self.bk_total_pnl = 0

    def reset(self) -> np.ndarray:
        """ Reset exchange status """
        print(f'Trading time is from {Timedelta(self._start_time, "ns")} to {Timedelta(self._end_time, "ns")}')
        self._path_pointer = (self._path_pointer + 1) % len(self._paths)
        self.tape = Tape(self._paths[self._path_pointer], latency=self._latency, end_time=self._end_time)

        self._open_positions.clear()
        self._position = 0
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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
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
        if action == 0:
            self._place_order(3, 3)
        elif action == 1:
            self._place_order(5, 2)
        elif action == 2:
            self._place_order(2, 5)
        elif action == 3:
            self._place_order(1, 1)  # For unittest
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
                if liquidation_pair:
                    # Assumes that both regular execution and liquidation hit the lower bound
                    return (
                        self._get_state(),
                        2 * self._reward_lb / 10000,
                        False,
                        {'pnl': reward_pair[1] + liquidation_pair[1]}
                    )
                return self._get_state(), reward_pair[0], True, {'pnl': reward_pair[1]}

            return self._get_state(), reward_pair[0], False, {'pnl': reward_pair[1]}
        return self._get_state(), 0, True, {'pnl': 0}

    def clean_up(self) -> None:
        final_mtm = self.book.mid_price

        while not self.tape.done:
            self._run_market()

        # Check if exchange finishes properly
        if not self.book.empty:
            raise RuntimeError('Market is not fully cleared')

        tmp = {idx: self.bk_action_counts[idx] for idx in range(self.action_space.n)}
        print(f'Actions: {tmp} | Bids: {self.bk_bid_counts} | Asks: {self.bk_ask_counts} | Cover: {self.bk_liquidation}'
              f' | Avg SP: {np.mean(self.bk_spread_profits) / 10000:.3g}'
              f' | SP: {np.sum(self.bk_spread_profits) / 10000}'
              f' | Pos PnL: {(self.bk_total_pnl + final_mtm * self._position) / 10000}'
              f' | Pos: {self._position}')

    def render(self, mode='human'):
        pass

    @property
    def position(self) -> int:
        return self._position

    # ========== Private methods ==========
    def _get_state(self) -> np.ndarray:
        """ Define the state of exchange """
        return np.array(sum((ind.update(self) for ind in self._indicators), ())).astype(np.float32)

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

    def _place_order(self, bid_dist: int, ask_dist: int) -> None:
        """
        * Basically, we only need a fancy UpdateOrder which place a LimitOrder at the specified price and cancel the
            original order if it exists when the UpdateOrder hit the market.
        * Sometime, the remaining order may be executed after the next action is issued and before the fancy LimitOrder
            hit the market, due to latency. This is fine. We just attribute the profit to the action that originates
            it.
        """
        # We try not to place order inside the market
        bid_depths, ask_depths = self.book.get_depth(num_levels=5)
        bid_price = bid_depths[bid_dist - 1][0]
        ask_price = ask_depths[ask_dist - 1][0]

        # "Fancy" LimitOrder will delete the existing one if it does exist when the new LimitOrder hits the market
        self.tape.add_user_order(UserLimitOrder(side='B', price=bid_price, shares=self._order_size))
        self.tape.add_user_order(UserLimitOrder(side='S', price=ask_price, shares=self._order_size))

    def _wait_for_execution(self) -> Optional[Tuple[float, float]]:
        """ If tape runs out before order is executed, None is returned """
        while not self.tape.done:
            execution = self._run_market()

            if execution:
                # Book keeping. Move booking to the front because _calculate_pnl will change Execution
                mid_price = self.book.mid_price
                self.bk_total_pnl -= mid_price * execution.shares
                self.bk_spread_profits.append((mid_price - execution.price) * execution.shares)
                if execution.shares > 0:
                    self.bk_bid_counts += 1
                else:
                    self.bk_ask_counts += 1

                # Update for current episode
                self._position += execution.shares

                # Derive reward
                pnl = self._calculate_pnl(execution)
                reward = min(self._reward_ub, max(self._reward_lb, pnl))

                return reward / 10000, pnl / 10000

            if self.tape.current_time > self._end_time:
                # If no execution and end time is passed, return
                return None

        return None

    def _calculate_pnl(self, execution: Execution) -> int:
        """ Calculate PnL on FIFO basis """
        pnl = 0
        if not self._open_positions or np.sign(execution.shares) == np.sign(self._open_positions[0].shares):
            self._open_positions.append(execution)
        else:
            while self._open_positions and abs(execution.shares) >= abs(self._open_positions[0].shares):
                matched_execution = self._open_positions.popleft()
                pnl += (execution.price - matched_execution.price) * matched_execution.shares
                execution.shares += matched_execution.shares

            if self._open_positions:
                pnl += (self._open_positions[0].price - execution.price) * execution.shares
                self._open_positions[0].shares += execution.shares
            elif execution.shares != 0:
                self._open_positions.append(execution)

        return pnl
