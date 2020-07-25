"""
Estimate sigma, A and k of Avellaneda market making model
"""
import pickle
import pandas as pd
import numpy as np

from rlmarket.market import OrderBook, LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder

# Parameters
file = 'AAPL_20170201'
delta = 60
gamma = 0.1

# Main Process
with open(f'../../data/parsed/{file}.pickle', 'rb') as f:
    tape = pickle.load(f)

# Set up time frame
current_time = int(pd.Timedelta('09:31:00').to_timedelta64())
time_delta = int(pd.Timedelta(seconds=delta).to_timedelta64())
end_time = int(pd.Timedelta('15:59:00').to_timedelta64())

# Get statistics
book = OrderBook()
mid_prices = []
bid_spreads = []
bid_intensities = []
ask_spreads = []
ask_intensities = []
spreads = []

for event in tape:
    if isinstance(event, LimitOrder):
        book.add_limit_order(event)
    elif isinstance(event, MarketOrder):
        book.match_limit_order(event)
    elif isinstance(event, CancelOrder):
        book.cancel_order(event)
    elif isinstance(event, DeleteOrder):
        book.delete_order(event)
    elif isinstance(event, UpdateOrder):
        book.modify_order(event)

    if current_time <= event.timestamp <= end_time:
        mid_prices.append(book.mid_price)
        spreads.append(book.spread)
        current_time += time_delta

assert book.empty
mid_prices = np.array(mid_prices) / 10000
std = np.std(mid_prices[1:] - mid_prices[:-1])
print(std * np.sqrt(len(mid_prices) - 1))
print(f'spread: {np.mean(spreads)}')
print(spreads[:20])
