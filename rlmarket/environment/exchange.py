from __future__ import annotations
import pickle
from typing import List, Union, Deque, Tuple, Optional, Dict
from collections import deque
from math import ceil, floor
from pandas import Timedelta

from rlmarket.market import OrderBook
from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder, Event
from rlmarket.market import UserLimitOrder, UserMarketOrder, Execution, UserEvent
from rlmarket.environment import Environment, StateT


class Tape:
    """ Provide efficient way to handle real order and user order flow """

    _curr_time: int = None

    def __init__(self, real_messages: List[Event], latency: int = 500000) -> None:
        self._real_queue = real_messages
        self._num_real_messages = len(real_messages)
        self._real_pointer = 0
        self._user_queue: Deque[UserEvent] = deque()
        self._delay = latency
        self._user_order_id = -1

    def add_user_order(self, order: UserEvent) -> int:
        """ Put user order on tape with correct timestamp and order ID """
        order.timestamp = self._curr_time + self._delay
        order.id = self._user_order_id
        self._user_order_id -= 1
        self._user_queue.append(order)
        return order.id

    def next(self) -> Union[Event, UserEvent]:
        """ Return the next order in sorted order """
        if self._user_queue and self._user_queue[0].timestamp < self._real_queue[self._real_pointer].timestamp:
            return self._user_queue.popleft()

        order = self._real_queue[self._real_pointer]
        self._curr_time = order.timestamp
        self._real_pointer += 1
        return order

    def reset(self):
        """ Reset tape to original state """
        self._real_pointer = 0
        self._user_order_id = -1
        self._user_queue.clear()

    @property
    def current_timestamp(self) -> int:
        return self._curr_time

    @property
    def done(self) -> bool:
        return self._real_pointer >= self._num_real_messages


class Exchange(Environment):
    """ Exchange emulate the electronic trading venue """

    def __init__(self, ticker: str, start_minute: int = 31) -> None:
        super().__init__(is_block_training=True)

        # Load real message data
        with open(f'data/parsed/{ticker}_20170201.pickle', 'rb') as f:
            self.tape = Tape(pickle.load(f))

        self.book = OrderBook()
        self.start_time = int(Timedelta(hours=9, minutes=start_minute).to_timedelta64())  # Start trading at say 9:31

        # User properties
        self.position = 0
        self.user_bids = 0  # Number of bid shares in order book
        self.user_asks = 0  # Number of ask shares in order book
        self.order_size = 50
        self.num_sub_episodes = 100  # Number of sub episodes to make up a full episode
        self.liquidation_ratio = 0.3  # The portion of position to neutralize when using MarketOrder
        self.rewards = []  # Store rewards for actions taking place in current episode
        self.reward_idx = 0  #
        self.reward_map: Dict[int, int] = {}  # Mapping from order ID to index of reward

    def reset(self) -> StateT:
        """ Reset exchange status """
        self.book.reset()
        self._preload_market()
        self.position = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[StateT, Optional[Union[float, List[float]]], bool]:
        """
        * We only ask for the next action after an order (limit / market) from the previous action is executed
        * We only update the agent every n steps. This should stabilizes the training a bit, like DQN
        * The execution of the last order is used as the break point to signal that the current episode is finished
        * The new action will place a new order on the side where last order is executed and will update the order
            on the side where the last order is not executed yet
        * We allow one order at a time
        """
        # Perform action
        if action < 5:
            # Symmetric market makding
            ids = self._place_order(action + 1, action + 1)
        elif action == 5:
            ids = self._place_order(3, 1)
        elif action == 6:
            ids = self._place_order(1, 3)
        elif action == 7:
            ids = self._place_order(5, 2)
        elif action == 8:
            ids = self._place_order(2, 5)
        elif action == 9:
            shares = int(self.position * self.liquidation_ratio)
            if self.position > 0:
                ids = self.tape.add_user_order(UserMarketOrder(side='S', shares=shares))
            elif self.position < 0:
                ids = self.tape.add_user_order(UserMarketOrder(side='B', shares=shares))
            else:
                raise RuntimeError('To be implemented')
        else:
            raise RuntimeError(f'Unrecognized action {action}')

        if isinstance(ids, int):
            ids = (ids,)

        for order_id in ids:
            self.reward_map[order_id] = self.reward_idx

        # Wait for result
        completed = False
        while not completed:
            executions = self._run_market()

            if executions:
                for exe in executions:
                    self.rewards[self.reward_map[exe.id]] += (self.book.mid_price - exe.price) * exe.shares

                    # We may get leftover response from the previous action. We only end the current sub-episode
                    #   when we get response for current action
                    if exe.id in ids:
                        completed = True

        self.reward_idx += 1
        state = self._get_state()
        done = self.tape.done
        if self.reward_idx >= self.num_sub_episodes or done:
            final_rewards = self.rewards
            self.rewards = [0] * self.num_sub_episodes
            self.reward_idx = 0
            return state, final_rewards, done

        return state, None, self.tape.done

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        """ To do later """

    @property
    def action_space(self) -> int:
        return 10

    @property
    def state_dimension(self) -> int:
        return 3

    # ========== Private Methods ==========
    def _get_state(self) -> StateT:
        """ Define the state of exchange """
        return self.book.mid_price, self.book.imbalance, self.position

    def _place_order(self, bid_dist: int, ask_dist: int) -> Tuple[int, int]:
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
        ask_price = ceil((mid_price - ask_dist * spread) / 100) * 100

        # "Fancy" LimitOrder will delete the existing one if it does exist when the new LimitOrder hits the market
        bid_id = self.tape.add_user_order(UserLimitOrder(side='B', price=bid_price, shares=self.order_size))
        ask_id = self.tape.add_user_order(UserLimitOrder(side='S', price=ask_price, shares=self.order_size))
        return bid_id, ask_id

    def _run_market(self) -> Optional[List[Execution]]:
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
            return [self.book.match_limit_order_for_user(order)]
        else:
            raise ValueError(f'Unrecognized order type {type(order)}')

    def _preload_market(self):
        """ Run the market until the trading start time """
        while True:
            self._run_market()
            if self.tape.current_timestamp >= self.start_time:
                break
