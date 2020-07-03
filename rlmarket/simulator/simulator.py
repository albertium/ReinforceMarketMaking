"""
Training orchestrator
"""
from rlmarket.agent import Agent
from rlmarket.market import Environment


class Simulator:
    """ Take in Agent and Environment and run training for a day and a ticker in episodes until converge """

    def __init__(self, agent: Agent, env: Environment) -> None:
        self.agent = agent
        self.env = env
        self.agent.set_num_actions(self.env.action_space)

    def train(self, n_iters: int = 100) -> None:
        """ Train agent repeatedly """

        for _ in range(n_iters):
            state = self.env.reset()
            done = False

            while not done:
                action = self.agent.act(state)
                new_state, reward, done = self.env.step(action)
                # Follow the convention of SARSA without the last A
                self.agent.update(state, action, reward, new_state)
                state = new_state
