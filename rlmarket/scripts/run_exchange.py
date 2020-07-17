
from rlmarket.simulator import BlockTrainer
from rlmarket.environment import Exchange
from rlmarket.environment.exchange_elements import MidPriceDeltaSign, Imbalance, Position
from rlmarket.environment.exchange_elements import RemainingTime, MidPrice, HalfSpread
from rlmarket.agent import TileCodingAgent, AvellanedaStoikovAgent

# indicators = [MidPriceDeltaSign(3), Imbalance(), Position()]
# agent = TileCodingAgent(alpha=0.001)
indicators = [RemainingTime(start_time=34200000000000, end_time=57600000000000), MidPrice(), HalfSpread(), Position()]
agent = AvellanedaStoikovAgent(sigma=1.3646, intensity=1)
env = Exchange('AAPL', start_time=34230000000000, indicators=indicators, order_size=50, position_limit=1000)

simulator = BlockTrainer(agent, env)
simulator.train(3)
