import time
import pickle

from rlmarket.market import OrderBook, LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder

with open('../../data/parsed/AAPL_20170201.pickle', 'rb') as f:
    tape = pickle.load(f)

start_time = time.time()
book = OrderBook()
for idx, event in enumerate(tape):
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

    # quote = book.quote
    # if quote[0] is not None and quote[1] is not None:
    #     print(quote)

assert book.quote == (None, None)
assert book.get_depth() == ([], [])
print(f'{len(tape)} records takes {time.time() - start_time:.1f}s')
