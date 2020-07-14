
from rlmarket.simulator import BlockTrainer
from rlmarket.environment import Exchange
from rlmarket.environment.exchange_elements import MidPriceDeltaSign, Imbalance, Position
from rlmarket.agent import TileCodingAgent

indicators = [MidPriceDeltaSign(3), Imbalance(), Position()]
env = Exchange('AAPL', start_time=34230000000000, indicators=indicators, order_size=50, position_limit=1000)
agent = TileCodingAgent()
simulator = BlockTrainer(agent, env)
simulator.train(2)
