"""
BlockTrainer updates agent by in blocks
"""
import pickle

from rlmarket.environment import Environment
from rlmarket.agent import Agent


class BlockTrainer:

    def __init__(self, agent: Agent, env: Environment, block_size: int, gamma: float) -> None:
        self.agent = agent
        self.env = env
        self.agent.set_num_states(self.env.state_dimension, self.env.action_space)
        self.decays = list(reversed([gamma ** i for i in range(block_size)]))

    def train(self, n_iterations: int = 100) -> None:
        """ Train agent repeatedly """
        n_rounds = 0
        skips = max(int(n_iterations / 100), 1)  # target to show 100 status
        tmp = {}

        for idx in range(n_iterations):
            state = self.env.reset()
            memory = []
            done = False
            cumulative_reward = 0
            cumulative_metric = 0

            # Main loop
            while not done:
                # Interacts with environment
                action = self.agent.act(state)
                new_state, reward, metric, done = self.env.step(action)
                memory.append((state, action, new_state))

                # Update agent
                if reward:
                    for (s, a, sp), decay in zip(memory, self.decays):
                        self.agent.update(s, a, decay * reward, sp)

                    # Update stats
                    n_rounds += min(len(memory), len(self.decays))
                    cumulative_reward += reward
                    cumulative_metric += metric
                    memory.clear()

                # Update state
                state = new_state

            self.env.clean_up()

            if idx % skips == skips - 1:
                print(f'Finished Iteration {idx} ({n_rounds}): {cumulative_reward:,.1f} / {cumulative_metric:,.1f}\n')

        with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/tile.pickle', 'wb') as f:
            pickle.dump(self.agent.q_function, f)

        with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/episodes.pickle', 'wb') as f:
            pickle.dump(tmp, f)
