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

# ====== Plot episodes ======
# episodes = {(k[0][0], k[1]): np.mean(v) for k, v in episodes.items()}
# states = sorted(set(k[0] for k in episodes))
# grid = []
# for state in states:
#     values = []
#     for action in range(9):
#         values.append(episodes[(state, action)])
#     grid.append(values)

# ====== Plot value function ======
grid = []
keys = sorted([k[0] for k in tile.table.keys()])
# for x in np.linspace(-3, 3, 50):
for x in keys:
    values = tile[(x,)]
    values = (values - np.mean(values)) / np.std(values)
    grid.append(values)

color_map = plt.imshow(grid, extent=[0, 8, 3, -3])
plt.colorbar(color_map)
plt.show()
