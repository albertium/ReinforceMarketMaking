"""
Training orchestrator
"""
import abc

from rlmarket.agent import Agent
from rlmarket.environment import Environment, StateT


class Trainer(abc.ABC):
    """ Take in Agent and Environment and run training for a day and a ticker in episodes until converge """

    def __init__(self, agent: Agent, env: Environment) -> None:
        self.agent = agent
        self.env = env
        self.agent.set_num_states(self.env.state_dimension, self.env.action_space)

    def train(self, n_iterations: int = 100) -> None:
        """ Train agent repeatedly """
        n_rounds = 0
        skips = int(n_iterations / 100)  # target to show 100 status

        for idx in range(n_iterations):
            state = self.env.reset()
            done = False
            total_reward = 0

            # Main loop
            while not done:
                n_rounds += 1
                state, reward, done = self._train_episode(state)
                total_reward += reward

            if idx % skips == skips - 1:
                print(f'Iteration {idx} ({n_rounds}): {total_reward}')

    @abc.abstractmethod
    def _train_episode(self, state: StateT):
        """ An episode has one update """
