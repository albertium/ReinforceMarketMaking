"""
Training orchestrator
"""
from collections import deque

from rlmarket.agent import Agent
from rlmarket.market import Environment


class Simulator:
    """ Take in Agent and Environment and run training for a day and a ticker in episodes until converge """

    def __init__(self, agent: Agent, env: Environment) -> None:
        self.agent = agent
        self.env = env
        self.agent.set_num_states(self.env.state_dimension, self.env.action_space)

    def train(self, n_iters: int = 100) -> None:
        """ Train agent repeatedly """
        memory = deque()
        is_last = False
        skips = int(n_iters / 100)  # target to show 100 status

        for idx in range(n_iters):
            state = self.env.reset()
            done = False
            total_rewards = 0

            # Last round is meant to be for display purpose. We record the moves and disable exploration
            if idx == n_iters - 1:
                is_last = True
                self.agent.disable_exploration()

            # Main loop
            while not done:
                action = self.agent.act(state)
                new_state, reward, done = self.env.step(action)
                self.agent.update(state, action, reward, new_state)

                # Record the last iteration
                if is_last:
                    memory.append((state, action, reward, new_state))

                state = new_state
                total_rewards += reward

            if idx % skips == skips - 1:
                print(f'Iteration {idx}: {total_rewards}')

        self.env.render(memory)
