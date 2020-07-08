# from rlmarket.utils import parse_raw_itch_file
#
#
# parse_raw_itch_file('AAPL', 'data/S010220-v50-bx.txt', 'data')

from rlmarket.utils import convert_csv_to_pickle, parse_raw_itch_file


for ticker in ['AAPL', 'AMZN', 'FB', 'MSFT', 'GOOG']:
    # parse_raw_itch_file(ticker, 'data/raw/S020117-v50.txt', 'data/parsed')
    convert_csv_to_pickle('data/parsed', f'{ticker}_20170201')