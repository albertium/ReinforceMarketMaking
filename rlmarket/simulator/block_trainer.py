from rlmarket.environment import StateT
from rlmarket.simulator.trainer import Trainer


class BlockTrainer(Trainer):

    def _train_episode(self, state: StateT):
        """ An episode in block mode consists of several sub-episode, which are only updated together at the end """
        memory = []
        while True:
            action = self.agent.act(state)
            new_state, reward, done = self.env.step(action)
            memory.append((state, action, new_state))
            state = new_state

            if reward or done:
                break

        # reward can be of length n - 1, in which case we do not update for the last episode
        if reward:
            for (state, action, new_state), r in zip(memory, reward[:-1]):
                self.agent.update(state, action, r, new_state)
            return state, reward[-1], done
        return state, 0, done
