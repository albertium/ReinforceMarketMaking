from collections import deque
from typing import Tuple, Deque
import plotly.graph_objects as go
import numpy as np

from rlmarket.environment import Environment, StateT


class SinePrice(Environment):

    def __init__(self, level: float = 100, amplitude: float = 2, cycle: int = 20, phase: float = 0,
                 lags: int = 5) -> None:

        self.level = level
        self.amplitude = amplitude
        self.cycle = cycle
        self.freq = 2 * np.pi / cycle
        self.phase = phase * 2 * np.pi
        self._step = 0
        self.lags = lags
        self.cache = deque()
        self.max_step = cycle * 5  # 5 cycles per training

    def generate_price(self, step: int) -> float:
        return self.level + self.amplitude * np.sin(self.freq * step + self.phase)

    def reset(self) -> StateT:
        self._step = 0
        self.cache.clear()
        for step in range(self.lags):
            self.cache.append(self.generate_price(step - self.lags + 1))
        return tuple(self.cache)

    def step(self, action: int) -> Tuple[StateT, float, bool]:
        """
        0 is short, 1 is flat, 2 is long.
        We assume position is executed before the new state is generated
        """
        self._step += 1

        # Mark new price and return
        new_price = self.generate_price(self._step)
        position = 2 * action - 1
        reward = position * (new_price - self.cache[-1])

        # Update cache
        self.cache.popleft()
        self.cache.append(new_price)

        return tuple(self.cache), reward, self._step >= self.max_step

    def render(self, memory: Deque[Tuple[StateT, int, float, StateT]]):
        longs = {'x': [], 'y': []}
        flats = {'x': [], 'y': []}
        shorts = {'x': [], 'y': []}
        for idx, (state, action, _, _) in enumerate(memory):
            if action == 0:
                shorts['y'].append(state[-1])
                shorts['x'].append(idx)
            elif action == 1:
                flats['y'].append(state[-1])
                flats['x'].append(idx)
            else:
                longs['y'].append(state[-1])
                longs['x'].append(idx)

        fig = go.Figure(data=[
            go.Scatter(x=longs['x'], y=longs['y'], fillcolor='green', name='Long', mode='markers'),
            go.Scatter(x=flats['x'], y=flats['y'], fillcolor='black', name='Flat', mode='markers'),
            go.Scatter(x=shorts['x'], y=shorts['y'], fillcolor='red', name='Short', mode='markers'),
        ])

        fig.write_html('abc.html')

    @property
    def action_space(self) -> int:
        """ 3 actions: short, flat, long """
        return 3

    @property
    def state_dimension(self) -> int:
        return self.lags

