"""
* Use tile coding for Q function. Can support continuous states
* Agent supports Q-learning and SARSA
"""
from rlmarket.agent.value_iteration_agent import ValueIterationAgent
from rlmarket.agent.q_function import TileCodedFunction


class TileCodingAgent(ValueIterationAgent):
    """ TD agent that uses tile coding for approximation """

    def initialize_q_function(self, state_dimension: int, num_actions: int) -> None:
        """
        We should use larger but more tiles so that local learning can be spread out to wider region.
        If instead we use finer but less tiles, we just end up with a fancy simple Q-table.
        """
        self.q_function = TileCodedFunction(state_dimension, num_actions, num_tiles=50, granularity=5)
