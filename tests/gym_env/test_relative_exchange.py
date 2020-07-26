"""
Unittest for rlmarket/gym_env/relative_exchange.py
"""
from rlmarket.gym_env import RelativeExchange
from rlmarket.market import Execution


def test_reward_calculation(mocker):
    """
    Focus on reward and pnl calculation since other are tested in AbsoluteExchange test already.
        * Case 1: first trade
        * Case 2: shares exactly matched
        * Case 3: shares less than the first in queue
        * Case 4: shares more than the first two in the queue
    """
    mocker.patch('rlmarket.environment.exchange_elements.pickle.load', return_value=[])
    mocker.patch('builtins.open', mocker.mock_open())

    position_limit = 100

    exchange = RelativeExchange(files=[''], indicators=[],
                                reward_lb=-0.02, reward_ub=0.02,
                                start_time=0, end_time=0,
                                latency=0, order_size=50, position_limit=position_limit)

    # Case 1 & 2
    assert exchange._calculate_reward(Execution(1, 10000, 100)) == (0, 0)
    assert exchange._calculate_reward(Execution(2, 10100, -100)) == (1, 1)

    # Case 3
    assert exchange._calculate_reward(Execution(3, 10000, 100)) == (0, 0)
    assert exchange._calculate_reward(Execution(4, 10500, -50)) == (1, 2.5)
    assert len(exchange._open_positions) == 1
    assert exchange._open_positions[0].shares == 50
    assert exchange._calculate_reward(Execution(5, 9500, -50)) == (-1.0, -2.5)

    # Case 4
    assert exchange._calculate_reward(Execution(6, 10000, -100)) == (0, 0)
    assert exchange._calculate_reward(Execution(7, 11000, -100)) == (0, 0)
    assert exchange._calculate_reward(Execution(8, 12000, -100)) == (0, 0)
    assert exchange._calculate_reward(Execution(9, 10000, 250)) == (5, 20)  # Pnl is 20 and return is more than 2%
