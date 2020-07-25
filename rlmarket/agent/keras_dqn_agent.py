"""
Adapt Keras DQN agent
"""
from typing import cast
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.optimizers import Adam

from rlmarket.agent import Agent
from rlmarket.environment import StateT


class KerasDQNAgent(Agent):
    """ Wrapper on Keras DQN agent """

    _internal_agent: DQNAgent

    def __init__(self) -> None:
        super().__init__()

    def set_num_states(self, state_dimension: int, num_actions: int) -> None:
        model = self._build_model(state_dimension, num_actions)
        memory = SequentialMemory(limit=10000, window_length=1)
        self._internal_agent = DQNAgent(model=model, nb_actions=num_actions, memory=memory,
                                        nb_steps_warmup=1000, target_model_update=1000,
                                        gamma=0.99, delta_clip=1)

        self._internal_agent.compile(Adam(lr=0.0001), metrics=['mae'])

    def act(self, state: StateT) -> int:
        return self._internal_agent.forward(state)

    def update(self, state: StateT, action: int, reward: float, new_state: StateT) -> None:
        self._internal_agent.backward()

    def _build_model(self, state_dimension: int, num_actions: int) -> Sequential:
        model = Sequential()
        model.add(Dense(units=64, input_shape=(1, state_dimension), activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_actions, activation='softmax'))
        return model
