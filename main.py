from rlmarket.utils import convert_csv_to_pickle, parse_raw_itch_file


# for ticker in ['SPY']:
#     # parse_raw_itch_file(ticker, 'data/raw/S020117-v50.txt', 'data/parsed')
#     convert_csv_to_pickle('data/parsed', f'{ticker}_20170201')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/tile.pickle', 'rb') as f:
    tile = pickle.load(f)

with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/episodes.pickle', 'rb') as f:
    episodes = pickle.load(f)

# with open('C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/episodes2.pickle', 'rb') as f:
#     transitions = pickle.load(f)

# ====== Plot episodes ======
# cache = {}
# for (s, a, sp), rs in episodes.items():
#     cache.setdefault((s, a), []).extend(rs)
#
# avg_rewards = {k: np.mean(v) for k, v in cache.items()}
# states = sorted(set(k[0] for k in avg_rewards))
# grid = []
# for state in states:
#     values = []
#     for action in range(3):
#         values.append(avg_rewards.get((state, action), 0))
#     print(f'{state[0]:.2f}', np.argmax(values))
#     grid.append(values)

# ====== Plot value function ======
# grid = []
# # keys = sorted([k[0] for k in tile.table.keys()])
# for x in np.linspace(-3, 3, 39):
#     values = tile[(x,)]
#     grid.append(values)
#     print(f'{x:.2f}', np.argmax(values))
#
#
# color_map = plt.imshow(grid, extent=[0, 8, 3, -3])
# plt.colorbar(color_map)
# plt.show()

import gym

env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.observation_space.shape)
