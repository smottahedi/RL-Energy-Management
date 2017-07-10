"""
Deep Double Q-learning Agent.
"""

import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import metrics
import numpy as np


class DDQNAgent(object):
    """
    Deep Double Q-Learning agent implementation.
    """


    def __init__(self, state_size, action_size,
                 memory_lenght=8000, discount=0.99, epsilon=1):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_lenght)
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()


    def _build_network(self):
        """
        build neural network for Deep Q-learning model.
        """

        network = Sequential()
        network.add(Dense(256, input_dim=self.state_size,
                         activation='tanh'))
        network.add(Dense(512, activation='tanh'))
        network.add(Dense(256, activation='tanh'))
        network.add(Dense(64, activation='tanh'))
        network.add(Dense(32, activation='tanh'))
        network.add(Dense(16, activation='tanh'))
        network.add(Dense(self.action_size, activation='linear'))
        network.compile(loss='mse',
                        metrics=[metrics.mse],
                       optimizer=Adam(lr=self.learning_rate))
        return network

    def update_target_network(self):
        # copy weights from model to target_model
        self.target_network.set_weights(self.network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        for mem in zip(state, action, reward, next_state, done):
            self.memory.append(mem)


    def act(self, states):
        actions = []
        for state in states:
            if np.random.rand() <= self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                action_value = self.network.predict(state.reshape((1, self.state_size)))
                actions.append(np.argmax(action_value))
        return actions

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.network.predict(state.reshape((1, self.state_size)))
            if done:
                target[0][action] = reward
            else:
                a = self.network.predict(next_state.reshape((1, self.state_size)))[0]
                t = self.target_network.predict(next_state.reshape((1, self.state_size)))[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.network.fit(state.reshape((1,self.state_size)), target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.network.load_weights(name)

    def save(self, name):
        self.network.save_weights(name)
