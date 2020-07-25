"""
Full example of stable baselines
"""
import numpy as np
from gym import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


class GridWorld(Env):

    def __init__(self, grid_size: int) -> None:
        self._grid_size = grid_size
        self._state = 0
        self.action_space = spaces.Discrete(2)  # Left or right
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(1,), dtype=np.float32)

    def step(self, action):
        if action == 0:
            self._state += 1
        elif action == 1:
            self._state -= 1
        else:
            raise ValueError(f'Unrecognized action {action}')

        self._state = np.clip(self._state, 0, self._grid_size - 1)
        done = bool(self._state == self._grid_size - 1)
        reward = 1 if done else 0
        return np.array([self._state]).astype(np.float32), reward, done, {}

    def reset(self):
        self._state = 0
        return np.array([self._state]).astype(np.float32)

    def render(self, mode='human'):
        pass


if __name__ == '__main__':
    check_env(GridWorld(10))
    env = make_vec_env(lambda: GridWorld(10), n_envs=1)

    model = PPO('MlpPolicy', env, verbose=1).learn(5000)

    state = env.reset()
    for _ in range(20):
        action, _ = model.predict(state, deterministic=True)
        # action = 0
        next_state, reward, done, info = env.step(action)
        print(f'{state} -> {action} -> {next_state}: {reward}')
        state = next_state
        if done:
            break
