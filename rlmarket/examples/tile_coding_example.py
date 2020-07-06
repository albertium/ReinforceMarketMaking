import numpy as np
import matplotlib.pyplot as plt

from rlmarket.agent import TileCodingAgent
from rlmarket.market import LinearWalk
from rlmarket.simulator import Simulator


agent = TileCodingAgent(alpha=0.01)
env = LinearWalk()
simulator = Simulator(agent, env)
simulator.train(500)

pos = np.linspace(-3, 3, 100)
y1 = tuple(agent.q_function[(x,)][0] for x in pos)
y2 = tuple(agent.q_function[(x,)][1] for x in pos)
plt.plot(pos, y1, label='left')
plt.plot(pos, y2, label='right')
plt.legend()
plt.show()
