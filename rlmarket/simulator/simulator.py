"""
Training orchestrator
"""
from typing import Tuple
from collections import deque

from rlmarket.agent import Agent
from rlmarket.environment import Environment, StateT


class Simulator:
    """ Take in Agent and Environment and run training for a day and a ticker in episodes until converge """

    def __init__(self, agent: Agent, env: Environment) -> None:
        self.agent = agent
        self.env = env
        self.agent.set_num_states(self.env.state_dimension, self.env.action_space)

    def _train_episode(self, state: StateT) -> Tuple[StateT, float, bool]:
        action = self.agent.act(state)
        new_state, reward, done = self.env.step(action)
        self.agent.update(state, action, reward, new_state)
        return new_state, reward, done

    def _train_blob(self, state: StateT) -> Tuple[StateT, float, bool]:
        """ A blob consist of several episodes but with just one update """
        memory = []
        while True:
            action = self.agent.act(state)
            new_state, reward, done = self.env.step(action)
            memory.append((state, action, new_state))

            if reward:
                break

        # reward can be of length n - 1, in which case we do not update for the last episode
        for (state, action, new_state), r in zip(memory, reward[:-1]):
            self.agent.update(state, action, r, new_state)
        return new_state, reward[-1], done

    def train(self, n_iters: int = 100) -> None:
        """ Train agent repeatedly """
        memory = deque()
        is_last = False
        n_rounds = 0
        skips = int(n_iters / 100)  # target to show 100 status

        for idx in range(n_iters):
            state: StateT = self.env.reset()
            done = False
            total_rewards = 0

            # Last round is meant to be for display purpose. We record the moves and disable exploration
            if idx == n_iters - 1:
                is_last = True
                self.agent.disable_exploration()

            # Main loop
            while not done:
                n_rounds += 1
                action = self.agent.act(state)
                new_state, reward, done = self.env.step(action)
                self.agent.update(state, action, reward, new_state)


            if idx % skips == skips - 1:
                print(f'Iteration {idx} ({n_rounds}): {total_rewards}')

        self.env.render(memory)
