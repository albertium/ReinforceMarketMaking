
from rlmarket.agent import ValueIterationAgent, DQNAgent
from rlmarket.market import Cliff
from rlmarket.simulator import Simulator


# agent = ValueIterationAgent(eps_next=0)
agent = DQNAgent(memory_size=4, batch_size=2, alpha=0.01)
env = Cliff()
sim = Simulator(agent, env)
sim.train(n_iters=1000)
