"""
The script show how to write a custom environment and agent
"""
import tensorflow as tf
from tensorflow import optimizers
import numpy as np
from tf_agents.environments import py_environment, tf_py_environment
from tf_agents.trajectories import time_step as ts, trajectory
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import tf_policy
from tf_agents.utils import common


class Cliff(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._state_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.int32, minimum=[0, 0], maximum=[3, 10],
                                                       name='state')

        self._state = np.zeros(2, dtype=np.int32)
        self._counter = 0
        self._done = False

    def observation_spec(self):
        return self._state_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._state = np.zeros(2, dtype=np.int32)
        self._counter = 0
        self._done = False
        return ts.restart(self._state)

    def _step(self, action):
        self._counter += 1
        if self._done:
            self.reset()

        if action == 0:
            if self._state[0] < 3:
                self._state[0] += 1
        elif action == 1:
            if self._state[0] > 0:
                self._state[0] -= 1
        elif action == 2:
            if self._state[1] < 10:
                self._state[1] += 1
        elif action == 3:
            if self._state[1] > 0:
                self._state[1] -= 1
        else:
            raise ValueError(f'Unrecognized action {action}')

        if self._counter >= 100:
            self._done = True
            return ts.termination(self._state, reward=-1)

        if self._state[0] == 0 and 1 <= self._state[1] <= 9:
            self._done = True
            return ts.termination(self._state, reward=-100)
        elif self._state[0] == 0 and self._state[1] == 10:
            self._done = True
            return ts.termination(self._state, reward=0)
        else:
            return ts.transition(self._state, reward=-1, discount=1.0)


def compute_average_reward(env: tf_py_environment.TFPyEnvironment, policy: tf_policy.Base, num_episodes=10) -> float:
    total_reward = 0
    for _ in range(num_episodes):
        time_step: ts.TimeStep = env.reset()
        episode_reward = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_reward += time_step.reward
            # print(action_step.action.numpy()[0], end=' ')
            print(time_step.observation.numpy())

        total_reward += episode_reward

    return total_reward / num_episodes


def collect_steps(env: tf_py_environment.TFPyEnvironment, policy: tf_policy.Base, buffer: ReplayBuffer):
    time_step = env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def train(num_iterations):
    train_env = tf_py_environment.TFPyEnvironment(Cliff())
    test_env = tf_py_environment.TFPyEnvironment(Cliff())
    counter = tf.Variable(0)

    # Build network
    network = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=(100,))
    agent = dqn_agent.DqnAgent(train_env.time_step_spec(), train_env.action_spec(), q_network=network,
                               optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
                               td_errors_loss_fn=common.element_wise_squared_loss, train_step_counter=counter)

    agent.initialize()
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                            batch_size=train_env.batch_size, max_length=100)
    dataset = buffer.as_dataset(sample_batch_size=32, num_steps=2)
    iterator = iter(dataset)
    first_reward = compute_average_reward(train_env, agent.policy, num_episodes=10)
    print(f'Before training: {first_reward}')
    rewards = [first_reward]

    for _ in range(num_iterations):
        for _ in range(2):
            collect_steps(train_env, agent.collect_policy, buffer)

        experience, info = next(iterator)
        loss = agent.train(experience).loss
        step_number = agent.train_step_counter.numpy()

        if step_number % 10 == 0:
            print(f'step={step_number}: loss={loss}')

        if step_number % 20 == 0:
            average_reward = compute_average_reward(test_env, agent.policy, 1)
            print(f'step={step_number}: Reward:={average_reward}')


if __name__ == '__main__':
    train(1000)
