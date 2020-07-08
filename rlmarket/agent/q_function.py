"""
Different forms of Q functions
"""
import abc
from typing import Tuple
from collections import defaultdict
from functools import partial

import numpy as np

from rlmarket.environment import StateT


class QFunction(abc.ABC):
    """ Base class for q_function types """

    @abc.abstractmethod
    def __getitem__(self, state: StateT) -> np.ndarray:
        """ Return q values of all actions under a state """

    @abc.abstractmethod
    def update(self, state: StateT, action: int, value: float, alpha: float):
        """ Update a state-action q values with learning rate alpha """


class SimpleQTable(QFunction):
    """ Simple Q-table supports discrete states """

    def __init__(self, num_actions: int) -> None:
        self.table = defaultdict(partial(np.zeros, num_actions))

    def __getitem__(self, state: StateT) -> np.ndarray:
        return self.table[state]

    def update(self, state: StateT, action: int, value: float, alpha: float):
        values = self.table[state]
        values[action] = (1 - alpha) * values[action] + alpha * value


class TileCodedFunction(QFunction):
    """
    Tile coded continuous states and approximate q-values with simple q-tables.
    Should not use on discrete values since that needs further fine-tuning.
    Also, discrete variable may not be ordinal. The concept of adjacency is important here
    """

    def __init__(self, num_actions: int, num_tiles: int, granularity: int) -> None:
        if granularity < 1:
            raise ValueError('Granularity should be at least 1')

        self.num_tiles = num_tiles
        self.tables = [SimpleQTable(num_actions) for _ in range(num_tiles)]

        # Features should be normalized already (mean = 0 and std = 1). This implicitly enforces stationarity
        base_bins = np.linspace(-3, 3, granularity + 1)
        dist = base_bins[1] - base_bins[0]
        base_bins -= dist * (num_tiles - 1) / num_tiles / 2
        shift = dist / num_tiles
        self.bin_groups = tuple(base_bins + shift * idx for idx in range(num_tiles))

    def __getitem__(self, state: StateT) -> np.ndarray:
        codes = self.encode_features(state)
        value = sum(table[code] for table, code in zip(self.tables, codes))
        return value / self.num_tiles

    def update(self, state: StateT, action: int, value: float, alpha: float):
        """ Update all active tiles with the same target values """
        codes = self.encode_features(state)
        for table, code in zip(self.tables, codes):
            table.update(code, action, value, alpha)

    def encode_features(self, state: StateT) -> Tuple[Tuple[int, ...], ...]:
        """ Convert continuous features into codes """
        return tuple(tuple(np.digitize(state, bins)) for bins in self.bin_groups)
