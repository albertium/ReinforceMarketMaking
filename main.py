from rlmarket.utils import convert_csv_to_pickle, parse_raw_itch_file


for ticker in ['SPY']:
    # parse_raw_itch_file(ticker, 'data/raw/S020117-v50.txt', 'data/parsed')
    convert_csv_to_pickle('data/parsed', f'{ticker}_20170201')

# import time
# import numpy as np
#
# decays = 1
# decays = np.array(decays)
# a = []
#
# start = time.time()
#
# for _ in range(100000):
#     if a:
#         b = 1
#
# print(f'{time.time() - start}')
