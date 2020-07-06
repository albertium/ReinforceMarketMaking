import numpy as np
import matplotlib.pyplot as plt

from rlmarket.agent import TileCodingAgent
from rlmarket.environment import LinearWalk, LinearWalk2D
from rlmarket.simulator import Simulator


agent = TileCodingAgent(alpha=0.01)
env = LinearWalk2D()
simulator = Simulator(agent, env)
simulator.train(1000)

# pos = np.linspace(-3, 3, 100)
# y1 = tuple(agent.q_function[(x,)][0] for x in pos)
# y2 = tuple(agent.q_function[(x,)][1] for x in pos)
# plt.plot(pos, y1, label='left')
# plt.plot(pos, y2, label='right')
# plt.legend()
# plt.show()

grid = []
for x in np.linspace(-3, 3, 50):
    values = []
    for y in np.linspace(-3, 3, 50):
        values.append(agent.q_function[x, y][0])
    grid.append(values)

color_map = plt.imshow(grid, extent=[-3, 3, 3, -3])
plt.colorbar(color_map)
plt.show()
