
from rlmarket.agent import SimpleTDAgent, TileCodingAgent
from rlmarket.market import Cliff
from rlmarket.simulator import Simulator

agent = SimpleTDAgent()
agent = TileCodingAgent()
# agent = DQNAgent(memory_size=4, batch_size=2, alpha=0.01)
env = Cliff()
sim = Simulator(agent, env)
sim.train(n_iters=1000)
