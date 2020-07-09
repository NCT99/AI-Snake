import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


class Agent:
    def __init__(self):
        self.memory = deque()
        self.gamma = 0.95
        self.epsilon = 0#1
        self.epsilon_decay = 0.99
        self.min_epsilon = 0#0.05
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self.neural_net()
        self.weights = 'weights/weights.h5'
        
    def neural_net(self):
        model = Sequential()
        model.add(Dense(activation="relu", input_dim=14, units=20))
        model.add(Dense(activation="relu", units=20))
        model.add(Dense(activation="relu", units=20))
        model.add(Dense(activation="linear", units=1))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def max_action_value(self, state):
        """
        We check the predicted state-action values for all 4 pairs
        and choose the best."""
        act_values = deque([])
        for i in range(4):
            state_action_pair = state.copy()
            state_action_pair.extend([i == e for e in range(4)])
            act_values.append(self.model.predict(np.array(state_action_pair).reshape((1, 14)))[0][0])
        return (np.argmax(np.array(act_values)),np.amax(np.array(act_values)))
        
    def action(self, state):
        if np.random.rand() <= max(self.epsilon, self.min_epsilon):
            return random.randrange(4)
        return self.max_action_value(state)[0]
    
    def memorize(self, state, action, reward, next_state, over):
        self.memory.append((state, action, reward, next_state, over))

    def learn(self, minibatch):
        for state, action, reward, next_state, over in minibatch:
            if not over:
                target = reward + self.gamma * self.max_action_value(next_state)[1]
            else:
                target = reward
            state_action_pair = state.copy()
            state_action_pair.extend([i == action for i in range(4)])
            self.model.fit(np.array(state_action_pair).reshape((1, 14)), np.array(target).reshape((1,1)), epochs=1, verbose=0)
    
    def load_weights(self):
        self.model.load_weights(self.weights)

    def save_weights(self):
        self.model.save_weights(self.weights)
    
