"""
Block training
"""

from rlmarket.simulator import Trainer
from rlmarket.environment import NewExchange
from rlmarket.environment.exchange_elements import MidPriceDeltaSign, Imbalance, Position, NormalizedPosition
from rlmarket.environment.exchange_elements import RemainingTime, MidPrice, HalfSpread
from rlmarket.agent import TileCodingAgent, AvellanedaStoikovAgent

position_limit = 1000
liquidation_ratio = 0.2
reward_lb = -120 * 4
reward_ub = 120 * 2

indicators = [MidPriceDeltaSign(3), Imbalance(), NormalizedPosition(position_limit)]
indicators = [NormalizedPosition(position_limit)]
agent = TileCodingAgent(alpha=0.05, warm_up_period=5000, eps_min=0.01, allow_exploration=True)

env = NewExchange(files=['AAPL_20170201'], indicators=indicators,
                  reward_lb=reward_lb, reward_ub=reward_ub,
                  start_time=34230000000000, end_time=57540000000000,
                  order_size=50, position_limit=position_limit, liquidation_ratio=liquidation_ratio)

simulator = Trainer(agent, env)
simulator.train(50)
