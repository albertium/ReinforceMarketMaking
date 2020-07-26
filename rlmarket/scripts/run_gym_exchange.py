"""
Use stable-baselines agent
"""
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.base_class import BaseAlgorithm

from rlmarket.gym_env import Exchange
from rlmarket.environment.exchange_elements import NormalizedPosition, MidPriceDeltaSign, Imbalance


def evaluate(environment: Exchange, model: BaseAlgorithm):
    state = environment.reset()
    total_pnl = 0
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = environment.step(action)
        total_reward += reward
        total_pnl += info['pnl']

    environment.clean_up()
    print(f'Pnl: {total_pnl} / Reward: {total_reward}')


if __name__ == '__main__':
    # Parameters
    order_size = 50
    position_limit = 1000
    liquidation_ratio = 0.2
    reward_lb = -120 * 4
    reward_ub = 120 * 2

    indicators = [MidPriceDeltaSign(3), Imbalance(), NormalizedPosition(position_limit)]

    env = Exchange(files=['AAPL_20170201'], indicators=indicators,
                   reward_lb=reward_lb, reward_ub=reward_ub,
                   start_time=34230000000000, end_time=57540000000000,
                   order_size=order_size, position_limit=position_limit, liquidation_ratio=liquidation_ratio)

    print('Checking environment')
    check_env(env)
    print('Done checking environment')

    # env = make_vec_env(lambda: env, n_envs=1)
    model = PPO('MlpPolicy', env, verbose=False)
    print('\nBegin training')
    for iteration in range(100):
        print(f'Iteration: {iteration}')
        model.learn(3000)
        evaluate(env, model)
