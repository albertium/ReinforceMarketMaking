"""
Training orchestrator
"""
import abc
from typing import Deque, Tuple
from collections import deque

from rlmarket.agent import Agent
from rlmarket.environment import Environment, StateT


class Trainer(abc.ABC):
    """ Take in Agent and Environment and run training for a day and a ticker in episodes until converge """

    def __init__(self, agent: Agent, env: Environment, lag: int = 0) -> None:
        self.agent = agent
        self.env = env
        self.queue_size = lag + 1
        self.agent.set_num_states(self.env.state_dimension, self.env.action_space)
        self.queue: Deque[Tuple[StateT, int, StateT]] = deque(maxlen=lag + 1)

    def train(self, n_iterations: int = 100) -> None:
        """ Train agent repeatedly """
        n_rounds = 0
        skips = max(int(n_iterations / 100), 1)  # target to show 100 status
        tmp = {}

        for idx in range(n_iterations):
            state = self.env.reset()
            self.queue.clear()
            done = False
            cumulative_reward = 0
            cumulative_metric = 0

            # Main loop
            while not done:
                # Interact with environment
                action = self.agent.act(state)
                new_state, reward, metric, done = self.env.step(action)

                # Update agent
                self.queue.append((state, action, new_state))
                if len(self.queue) == self.queue_size:
                    s, a, sp = self.queue.popleft()
                    self.agent.update(s, a, reward, sp)
                    tmp.setdefault((s, a, sp), []).append(reward)

                # Update state
                state = new_state

                # Update stats
                n_rounds += 1
                cumulative_reward += reward
                cumulative_metric += metric

            self.env.clean_up()

            if idx % skips == skips - 1:
                print(f'Finished Iteration {idx} ({n_rounds}): {cumulative_reward:,.1f} / {cumulative_metric:,.1f}\n')

            if idx % 5 == 4:
                # Do a evaluation round
                self.agent.go_greedy()
                state = self.env.reset()
                done = False
                cumulative_reward = 0
                cumulative_metric = 0

                # Main loop
                while not done:
                    # Interact with environment
                    action = self.agent.act(state)
                    state, reward, metric, done = self.env.step(action)

                    # Update stats
                    cumulative_reward += reward
                    cumulative_metric += metric

                self.agent.go_normal()
                self.env.clean_up()
                print(f'==== Evaluation ==== {cumulative_reward:,.1f} / {cumulative_metric:,.1f}\n')

        import pickle
        with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/tile.pickle', 'wb') as f:
            pickle.dump(self.agent.q_function, f)

        with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/episodes.pickle', 'wb') as f:
            pickle.dump(tmp, f)
