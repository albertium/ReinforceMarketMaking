"""
Exchange environment based on gym env
"""
from typing import Tuple
import numpy as np

from rlmarket.gym_env.base_exchange import BaseExchange
from rlmarket.market import Execution


class RelativeExchange(BaseExchange):
    """ Exchange environment that implements the gym env API """

    def _calculate_reward(self, execution: Execution) -> Tuple[float, float]:
        """ Calculate PnL on FIFO basis and set reward as percentage """
        shares = execution.shares
        open_mtm = 0
        if not self._open_positions or np.sign(execution.shares) == np.sign(self._open_positions[0].shares):
            self._open_positions.append(execution)
            return 0, 0
        else:
            while self._open_positions and abs(execution.shares) >= abs(self._open_positions[0].shares):
                matched_execution = self._open_positions.popleft()
                open_mtm -= matched_execution.price * matched_execution.shares
                execution.shares += matched_execution.shares

            if self._open_positions:
                open_mtm += self._open_positions[0].price * execution.shares
                self._open_positions[0].shares += execution.shares
            elif execution.shares != 0:
                shares -= execution.shares
                self._open_positions.append(execution)

        pnl = (-execution.price * shares + open_mtm)
        reward = min(self._reward_ub, max(self._reward_lb, pnl / abs(open_mtm))) * abs(shares)
        return reward, pnl / 10000
