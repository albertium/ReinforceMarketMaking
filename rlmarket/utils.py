import struct
from pathlib import Path
import pickle
from datetime import datetime

from rlmarket.market import LimitOrder, MarketOrder, CancelOrder, DeleteOrder, UpdateOrder


def parse_raw_itch_file(ticker, infile, out_dir):
    name = Path(infile).stem
    if 'v50' not in name:
        raise ValueError('not a ITCH 5 file')

    date_string = datetime.strftime(datetime.strptime(name.split('-')[0], 'S%m%d%y'), '%Y%m%d')

    with open(infile, "rb") as f:
        data = f.read()  # Read the whole file into memory, assuming the machine has sufficient memory
        idx = 0  # Pointer to the current byte
        output = []
        record = {}  # save side of limit orders
        target = None

        # Find stock locate
        # Format: size|code|msg
        while idx < len(data) + 1:
            size = data[idx + 1]
            idx += 2 + size
            msg = data[idx - size: idx]
            if msg[0] == 82:
                locate, tracking, _, timestamp, stock, exchange, status, lot = struct.unpack("!HHHI8sccl", msg[1:25])
                if stock.decode().strip() == ticker:
                    target = locate
                    print(f'Ticker {ticker} found')
                    print(f'exchange:   {exchange.decode()}')
                    print(f'stock:      {ticker}')
                    print(f'locate:     {target}')
                    print(f'lot size:   {lot}')
                    break

            if msg[0] == 65:
                break  # already pass the stock directory section

        if target is None:
            print(f'stock {ticker} not found')
            return

        # convert limit order side to environment order side
        reverse_map = {'B': 'S', 'S': 'B'}

        # parse and find messages of ticker
        while idx < len(data) + 1:
            size = data[idx + 1]
            idx += 2 + size
            if idx >= len(data):
                break
            msg = data[idx - size: idx]
            msg_type = chr(msg[0])

            if msg_type == 'A' or msg_type == 'F':
                locate, tracking, timestamp, ref, buy_sell, shares, stock, price = struct.unpack("!HH6sQcI8sI",
                                                                                                 msg[1: 36])
                if locate == target:
                    timestamp = int.from_bytes(timestamp, "big")
                    buy_sell = buy_sell.decode()
                    record[ref] = buy_sell
                    output.append(['A', timestamp, ref, buy_sell, price, shares])
            elif msg_type == 'E' or msg_type == 'C':
                locate, tracking, timestamp, ref, shares, match = struct.unpack("!HH6sQIQ", msg[1:31])
                if locate == target:
                    timestamp = int.from_bytes(timestamp, "big")
                    output.append(['E', timestamp, ref, reverse_map[record[ref]], shares])
            elif msg_type == 'X':
                locate, tracking, timestamp, ref, shares = struct.unpack("!HH6sQI", msg[1:])
                if locate == target:
                    timestamp = int.from_bytes(timestamp, "big")
                    output.append(['X', timestamp, ref, shares])
            elif msg_type == 'D':
                locate, tracking, timestamp, ref = struct.unpack("!HH6sQ", msg[1:])
                if locate == target:
                    timestamp = int.from_bytes(timestamp, "big")
                    del record[ref]
                    output.append(['D', timestamp, ref])
            elif msg_type == 'U':
                locate, tracking, timestamp, ref, new_ref, shares, price = struct.unpack("!HH6sQQII", msg[1:])
                if locate == target:
                    timestamp = int.from_bytes(timestamp, "big")
                    record[new_ref] = record[ref]
                    del record[ref]
                    output.append(['U', timestamp, new_ref, ref, price, shares])

    with open(Path(out_dir) / f'{ticker}_{date_string}.csv', "w") as f:
        text = [",".join([str(elem) for elem in row]) for row in output]
        f.write("\n".join(text) + "\n")


def convert_csv_to_pickle(path, file):
    full_path = Path(path) / file
    queue = []
    with open(full_path.with_suffix('.csv'), 'r') as f:
        msgs = f.readlines()
        for msg in msgs:
            msg = msg.split(',')
            timestamp = int(msg[1])
            ref = int(msg[2])

            if msg[0] == 'A':
                order = LimitOrder(timestamp, ref, msg[3], int(msg[4]), int(msg[5]))
            elif msg[0] == 'E':
                order = MarketOrder(timestamp, ref, msg[3], int(msg[4]))
            elif msg[0] == 'X':
                order = CancelOrder(timestamp, ref, int(msg[3]))
            elif msg[0] == 'D':
                order = DeleteOrder(timestamp, ref)
            elif msg[0] == 'U':
                order = UpdateOrder(timestamp, ref, int(msg[3]), int(msg[4]), int(msg[5]))
            else:
                raise ValueError('Unknown order type')
            queue.append(order)

    with open(full_path.with_suffix('.pickle'), 'wb') as f:
        pickle.dump(queue, f)
