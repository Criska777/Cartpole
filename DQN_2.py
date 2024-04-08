import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 300
    batch_size = 64

    scores = []  # List to save scores from each episode for plotting
    epsilons = []  # List to save epsilon values for plotting
    times = []  # List to save times (length of each episode) for plotting

    for e in range(episodes):
        state = env.reset()
        state_array = state[0]
        state_list = state_array.tolist()
        state = np.array(state_list).flatten()  # Flatten state to ensure it is a 1D array
        state = np.reshape(state, [1, state_size])  # Reshape for neural network
        score = 0

        for time in range(500):  # Set max time steps per episode
            action = agent.act(state)
            next_state, reward, done, _ , _ = env.step(action)
            reward = reward if not done else -10  # penalize for dropping pole
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                agent.update_target_model()
                print(f"Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2}")
                scores.append(score)  # Save score for plotting
                epsilons.append(agent.epsilon)  # Save epsilon for plotting
                times.append(time)  # Save time (length of episode) for plotting
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    # Plot the scores, epsilon values, and times after training
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.ylabel('Epsilon')

    plt.subplot(3, 1, 2)
    plt.plot(times)
    plt.title('Time per Episode')
    plt.ylabel('Time')

    plt.subplot(3, 1, 3)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.ylabel('Score')
    plt.xlabel('Episode')

    plt.tight_layout()
    plt.show()
