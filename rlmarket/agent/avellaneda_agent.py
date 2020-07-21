"""
Non-learning agent that execute Avellaneda Stoikov market making strategy
"""
import math

from rlmarket.agent import Agent
from rlmarket.environment import StateT


class AvellanedaStoikovAgent(Agent):

    actions = {
        0: (1, 1),
        1: (2, 2),
        2: (3, 3),
        3: (4, 4),
        4: (5, 5),
        5: (3, 1),
        6: (1, 3),
        7: (5, 2),
        8: (2, 5),
    }

    def __init__(self, sigma: float, gamma: float = 0.1, intensity: float = 1.5) -> None:
        # Prepare constants
        self.scaled_variance = gamma * sigma ** 2
        self.nudge = math.log(1 + gamma / intensity) / gamma

    def act(self, state: StateT) -> int:
        # Calculate model bid and ask
        remaining_time, mid_price, book_half_spread, position = state
        total_variance = self.scaled_variance * remaining_time
        price = mid_price - total_variance * position
        half_spread = total_variance / 2 + self.nudge
        bid, ask = price - half_spread, price + half_spread

        # Convert to allowed actions
        bid_dist = min(5, max(1, round((mid_price - bid) / book_half_spread)))
        ask_dist = min(5, max(1, round((ask - mid_price) / book_half_spread)))

        selected_code = -1
        distance = 10
        for action_code, (bd, ad) in self.actions.items():
            new_distance = abs(bid_dist - bd) + abs(ask_dist - ad)
            if new_distance < distance:
                distance = new_distance
                selected_code = action_code

        return selected_code

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        """ Do not need to set number of states because we are not really learning """

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        """ No need to update because we are not really learning """

    def go_greedy(self):
        """ Not applicable because we are not really learning """
