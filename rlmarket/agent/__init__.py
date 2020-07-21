"""
Reference:
* https://spinningup.openai.com/en/latest/index.html#
* https://medium.com/@SmartLabAI/reinforcement-learning-algorithms-an-intuitive-overview-904e2dff5bbc
    An introduction based on the first link

Roughly
* Model-free RL: no internal model, learned values completely based on experience
    * Policy Iteration:
        1. Directly learn the mapping from state to action. Policy is learned without value function
        2. Values are derived based on the policy
        3. There are deterministic policy and stochastic policy

        * Policy Gradient (PG):
            1. pi_theta (a|s) = P(a|s)
            2. theta is parameters. The policy is a conditional distribution
            3. We want to maximize the expected sum of discounted rewards with respect to theta

        * Asynchronous Advantage Actor-Critic (A3C):
            1. Asynchronous: multiple agents are trained independently at the same time
            2. Advantage: update rule is based on sum of discounted rewards
            3. Actor-Critic: learn both policy (using PG for example) and value function (using value iteration for
                example)

        * Trust Region Policy Optimization (TRPO)
            1. Maximize policy with distance constraint to the previous policy

        * Proximal Policy Optimization (PPO)
            1. Later

    * Value Iteration
        1. Directly learn the action-value function (Q function) or value function (V function)
        2. Policy is derived from the Q function

        * Temporal-Difference (TD):
            1. The idea is to make V(S(t)) approach R(t+1) + gamma * V(S(t+1))
            2. That is, V(S(t)) += alpha * (R(t+1) + gamma * V(S(t+1)) - V(S(t)))
            3. Similarly with Q function, Q(s(t), a(t)) += alpha * (R(t+1) + gamma * Q(s(t+1), a(t+1)) - Q(s(t), a(t)))

        * SARSA (On-policy TD):
            1. Compared to pure TD, we need to derive policy at each step
            2. We achieve this by picking the action that maximize Q value at the given state (usually e-greedy)
            3. Pure TD will pick action uniformly?
            4. In general, SARSA performs better than Q-learning below because we are free to choose more robust policy
                at S(t+1)

        * Q-learning (Off-policy TD):
            1. Off-policy in the sense that the policy used in S(t) may not be the same as that used in S(t+1)
            2. At S(t), we generally use e-greedy, but at S(t+1), Q-learning uses max
            3. Therefore, Q-learning and SARSA are the same if the policy at S(t) is max
            4. In this sense, SARSA is a generalization of Q-learning

        * Deep Q Network (DQN):
            1. Use deep network to represent the Q function
            2. However, Q-learning is unstable in this setting (non-linear Q function) and DQN is designed to stabilize
                Q-learning
            3. Experience replay: episodes (a full SARSA step) are cached and update is done on a sampled episode.
                This is to remove sequential correlation and smooth data distribution
            4. Network is only updated after every n steps

* Model-based RL: later
"""
from rlmarket.agent.base_agent import Agent
from rlmarket.agent.simple_td_agent import SimpleTDAgent
from rlmarket.agent.tile_coding_agent import TileCodingAgent
from rlmarket.agent.tile_coding_memory_agent import TileCodingMemoryAgent
from rlmarket.agent.dqn_agent import DQNAgent
from rlmarket.agent.avellaneda_agent import AvellanedaStoikovAgent
