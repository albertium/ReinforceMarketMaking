from rlmarket.agent import TileCodingAgent
from rlmarket.environment import SinePrice
from rlmarket.simulator import Simulator

agent = TileCodingAgent(warm_up_period=10000, alpha=0.001)
# agent = DQNAgent(memory_size=4, batch_size=2, alpha=0.01)
env = SinePrice(lags=8, level=0, amplitude=3, cycle=100)
sim = Simulator(agent, env)
sim.train(n_iters=100)
