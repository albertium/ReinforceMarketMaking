
from rlmarket.simulator import Trainer
from rlmarket.environment import Exchange
from rlmarket.environment.exchange_elements import MidPriceDeltaSign, Imbalance, Position, NormalizedPosition
from rlmarket.environment.exchange_elements import RemainingTime, MidPrice, HalfSpread
from rlmarket.agent import TileCodingAgent, AvellanedaStoikovAgent, SimpleTDAgent, TileCodingMemoryAgent


position_limit = 1000
liquidation_ratio = 0.2

indicators = [MidPriceDeltaSign(3), Imbalance(), NormalizedPosition(position_limit)]
indicators = [NormalizedPosition(position_limit)]
agent = TileCodingAgent(alpha=0.1, warm_up_period=5000, eps_min=0.01, allow_exploration=False)
agent = TileCodingMemoryAgent(alpha=0.01, warm_up_period=5000, eps_min=0.01, allow_exploration=False,
                              memory_size=1000, batch_size=10)

# agent = SimpleTDAgent(alpha=0.1, eps_min=0.01, warm_up_period=5000, allow_exploration=False)
# indicators = [RemainingTime(start_time=34200000000000, end_time=57600000000000), MidPrice(), HalfSpread(), Position()]
# agent = AvellanedaStoikovAgent(sigma=1.3646, intensity=1)

env = Exchange(files=['AAPL_20170201'], indicators=indicators,
               start_time=34230000000000, end_time=57540000000000, order_size=50,
               position_limit=position_limit, liquidation_ratio=liquidation_ratio)

simulator = Trainer(agent, env, lag=1)
simulator.train(50)
