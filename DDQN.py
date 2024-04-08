import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import pickle

class DQNAgent:
    def __init__(self, state_size, action_size,
                 memory_size=2000,
                 batch_size=48,
                 discount_factor=0.9,
                 learning_rate=0.001,
                 train_start=1000,
                 epsilon=1.0,
                 epsilon_decay_rate=0.99,
                 ddqn_flag=True,
                 polyak_avg=False,
                 pa_tau=0.1,
                 duel_flag=False,
                 dueling_option='avg',
                 load_weights_path=None,
                 load_exp_path=None):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.train_start = train_start
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.ddqn = ddqn_flag
        self.dueling_option = dueling_option
        self.polyak_avg = polyak_avg
        self.pa_tau = pa_tau

        self.model = self._build_model()
        self.target_model = self._build_model()

        if load_weights_path is not None:
            self.model.load_weights(load_weights_path)
            print('Weights loaded from File')

        if load_exp_path is not None:
            with open(load_exp_path, 'rb') as file:
                self.memory = pickle.load(file)
                print('Experience loaded from file')

        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        network_input = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(network_input)
        x = Dense(24, activation='relu')(x)
        network_output = Dense(self.action_size, activation='linear')(x)

        model = Model(network_input, network_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def update_target_network(self):
        if self.ddqn and self.polyak_avg:
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = weights[i] * self.pa_tau + target_weights[i] * (1 - self.pa_tau)
            self.target_model.set_weights(target_weights)
        else:
            self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay_rate
        else:
            self.epsilon = 0.01

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.model.predict(state)[0])

    def get_maxQvalue_nextstate(self, next_state):
        if self.ddqn:
            action = np.argmax(self.model.predict(next_state)[0])
            max_q_value = self.target_model.predict(next_state)[0][action]
        else:
            max_q_value = np.amax(self.target_model.predict(next_state)[0])
        return max_q_value

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        current_state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        qValues = np.zeros((self.batch_size, self.action_size))

        for i in range(self.batch_size):
            current_state[i] = mini_batch[i][0]
            action = mini_batch[i][1]
            reward = mini_batch[i][2]
            next_state[i] = mini_batch[i][3]
            done = mini_batch[i][4]

            qValues[i] = self.model.predict(current_state[i].reshape(1, self.state_size))[0]
            max_qvalue_ns = self.get_maxQvalue_nextstate(next_state[i].reshape(1, self.state_size))

            if done:
                qValues[i][action] = reward
            else:
                qValues[i][action] = reward + self.discount_factor * max_qvalue_ns

        self.model.fit(current_state, qValues, batch_size=self.batch_size, epochs=1, verbose=0)
        self.update_epsilon()

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

deepQ = DQNAgent(state_size, action_size)

max_episodes = 1000
targetNetworkUpdateFreq = 10
train_start = 1000
model_save_freq = 300

last100Scores = deque(maxlen=100)
Scores = []
AvgScores = []
Avg100Scores = []

for e in range(max_episodes):
    state = env.reset()
    state_array = state[0]
    state_list = state_array.tolist()
    state = np.array(state_list).flatten()
    state = np.reshape(state, [1, state_size])
    t = 0
    done = False
    while not done:
        action = deepQ.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        reward = reward if not done else -100

        t += 1

        deepQ.add_experience(state, action, reward, next_state, done)
        deepQ.train_model()

        state = next_state

        if done:
            deepQ.update_target_network()

            Scores.append(t)
            AvgScores.append(np.mean(Scores))
            last100Scores.append(t)
            Avg100Scores.append(np.mean(last100Scores))

            print('Episode: {}, time steps: {}, AvgScore: {:0.2f}, epsilon: {:0.2f}, replay size: {}'.format(e, t, np.mean(Scores), deepQ.epsilon, len(deepQ.memory)))

    if np.mean(last100Scores) > (env.spec.max_episode_steps - 5):
        print('The problem is solved in {} episodes. Exiting'.format(e))
        break

env.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(deepQ.epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay')

plt.subplot(1, 3, 2)
plt.plot(Scores)
plt.xlabel('Episode')
plt.ylabel('Time Steps')
plt.title('Time Steps per Episode')

plt.subplot(1, 3, 3)
plt.plot(AvgScores)
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Average Score per Episode')

plt.tight_layout()
plt.show()
