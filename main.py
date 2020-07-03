
from rlmarket.agent import ValueIterationAgent
from rlmarket.market import GridWorld
from rlmarket.simulator import Simulator


agent = ValueIterationAgent(eps_now=1)
sim = Simulator(agent, GridWorld(nrows=3, ncols=3, start=(0, 0), end=(2, 2), hole=(0, 1)))
sim.train(n_iters=1000)
for i in range(3):
    for j in range(3):
        print((i, j), agent.q_function[(i, j)])

