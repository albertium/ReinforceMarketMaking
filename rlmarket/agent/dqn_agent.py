import torch
import random
import numpy as np

from rlmarket.agent import Agent
from rlmarket.environment import StateT


class ReplayMemory:
    """ Store episodes for DQN training """

    def __init__(self, memory_size: int, batch_size: int):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, state: StateT, action: int, reward: float, new_state: StateT):
        """ Saves a transition. """
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, new_state)
        self.position = (self.position + 1) % self.memory_size

    def sample(self):
        """ Sample past episodes for training """
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent):

    num_actions: int
    policy_network: torch.nn.Module
    target_network: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, eps_decay: float = 0.995, min_eps_curr: float = 0.05, min_eps_next: float = 0.05,
                 alpha: float = 1e-5, gamma: float = 0.99, tau: float = 0.001,
                 ratio: int = 2, memory_size: int = 10000, batch_size: int = 128) -> None:

        # Nothing is learned at the beginning. We should use totally random policy
        self.eps_decay = eps_decay
        self.eps_curr, self.min_eps_curr = 1, min_eps_curr
        self.eps_next, self.min_eps_next = 1, min_eps_next
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        # Network parameters
        self.ratio = ratio
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size, batch_size)
        self.loss_fn = torch.nn.MSELoss()

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        self.num_actions = num_actions

        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, self.ratio * state_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ratio * state_dimension, num_actions),
        )

        self.target_network = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, self.ratio * state_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ratio * state_dimension, num_actions),
        )

        # Initialize target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Set up optimizer
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.alpha)

    def act(self, state: StateT) -> int:
        if random.random() < self.eps_curr:
            return np.random.randint(self.num_actions)
        return self.policy_network(torch.tensor((state,), dtype=torch.float)).argmax().item()

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        """
        * Update memory.
        * Update policy network. Use policy network for current state and target network for next state
        * Soft update target network
        """
        self.memory.push(state, action, reward, new_state)

        if len(self.memory) < self.batch_size:
            return

        # Organize input
        episodes = self.memory.sample()
        states, actions, rewards, new_states = tuple(zip(*episodes))

        # Convert to tensors
        states, new_states = torch.tensor(states, dtype=torch.float), torch.tensor(new_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64).reshape(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float)

        # Update Q-values
        q_values_per_action = self.policy_network(states).gather(1, actions)

        # "detach" to remove next_q_values from graph. We only use next_q_values for values.
        next_q_values = self.target_network(new_states).max(1)[0].detach()
        target_q_values = (self.gamma * next_q_values + rewards).reshape(-1, 1)
        loss = self.loss_fn(q_values_per_action, target_q_values)

        print(f'States: {states}')
        print(f'Actions: {actions}')
        print(f'network: {q_values_per_action}')
        print(f'target: {target_q_values}')
        print(self.policy_network(states))

        # Update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(self.policy_network(states))
        print()

        # Soft update target network
        policy_state = self.policy_network.state_dict()
        target_state = self.target_network.state_dict()
        for state, values in policy_state.items():
            target_state[state] = (1 - self.tau) * target_state[state] + self.tau * values
        self.target_network.load_state_dict(target_state)

        # Update e-greedy threshold
        if self.eps_curr > self.min_eps_curr:
            self.eps_curr *= self.eps_decay

    def disable_exploration(self):
        self.eps_curr = 0
        self.eps_next = 0
