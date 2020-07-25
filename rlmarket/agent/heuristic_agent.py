"""
Agent based on heuristic rules
"""
from rlmarket.agent import Agent
from rlmarket.environment import StateT


class HeuristicAgent(Agent):
    """ Agent only acts on action 2, 7, and 8 based on heuristic rules """

    def __init__(self, decision_boundary: int) -> None:
        self.decision_boundary = decision_boundary

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        pass

    def act(self, state: StateT) -> int:
        position = state[0]
        if abs(position) > self.decision_boundary:
            if position > 0:
                return 1  # Quote at (5, 2)
            else:
                return 2  # Quote at (2, 5)
        return 0  # Quote at (3, 3)

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        pass

    def go_greedy(self):
        pass

    def go_normal(self):
        pass
