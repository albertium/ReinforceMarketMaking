from rlmarket.utils import convert_csv_to_pickle, parse_raw_itch_file

root = 'C:/Users/Albert/PycharmProjects/ReinforceMarketMaking/'

for ticker in ['SPY', 'AAPL', 'AMZN', 'FB', 'GOOG', 'MSFT']:
    # parse_raw_itch_file(ticker, root + 'data/raw/S020317-v50.txt', root + 'data/parsed')
    convert_csv_to_pickle(root + 'data/parsed', f'{ticker}_20170203')