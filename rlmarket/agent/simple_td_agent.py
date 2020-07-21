"""
* Simple q function that only supports discrete state
* Q-learning or SARSA agent
"""

from rlmarket.agent.value_agent import ValueIterationAgent
from rlmarket.agent.q_function import SimpleQTable


class SimpleTDAgent(ValueIterationAgent):
    """ Agent that uses TD to learning q-values """

    def initialize_q_function(self, state_dimension: int, num_actions: int) -> None:
        self.q_function = SimpleQTable(num_actions)
