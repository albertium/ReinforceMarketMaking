"""
Test ValueIterationAgent
"""
from rlmarket.agent import SimpleTDAgent


def test_value_iteration_agent():
    """ Test the Q-value update of the agent using GridWorld like example """
    agent = SimpleTDAgent(eps_curr=1)
    agent.set_num_states(2, 3)
    actions = [agent.act((0, 0)) for _ in range(20)]

    # Check e-greedy
    assert 0 in actions
    assert 1 in actions
    assert 2 in actions

    # Check update
    agent = SimpleTDAgent(eps_curr=0, eps_next=0, gamma=0.5)  # disable e-greedy
    agent.set_num_states(2, 3)
    agent.update((0, 0), 1, 1, (0, 1))
    assert agent.q_function[(0, 0)][1] == 0.3
    agent.update((0, 1), 2, 1, (0, 0))
    assert agent.q_function[(0, 1)][2] == 0.345
    agent.update((0, 0), 1, -1, (0, 1))
    assert agent.q_function[(0, 0)][1] == 0.3 * (-1 + 0.5 * 0.345) + 0.7 * 0.3  # -0.03825
    agent.update((0, 1), 0, 2, (0, 0))
    assert agent.q_function[(0, 1)][0] == 0.3 * 2
