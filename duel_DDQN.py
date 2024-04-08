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

class DQNAgent:  # Define DQNAgent class
    def __init__(self, state_size, action_size,  # Initialization method
                 memory_size=2000,  # Set experience replay memory size
                 batch_size=48,  # Set batch size
                 discount_factor=0.9,  # Set discount factor
                 learning_rate=0.001,  # Set learning rate
                 train_start=1000,  # Set start training step
                 epsilon=1.0,  # Set initial exploration rate
                 epsilon_decay_rate=0.99,  # Set exploration rate decay rate
                 ddqn_flag=True,  # Set whether to use double deep Q network
                 polyak_avg=False,  # Set whether to use Polyak averaging
                 pa_tau=0.1,  # Set Polyak averaging weight
                 duel_flag=True,  # Set whether to use Dueling architecture
                 dueling_option='avg',  # Set Dueling architecture type
                 load_weights_path=None,  # Set path to load weights
                 load_exp_path=None):  # Set path to load experience

        self.state_size = state_size  # Set state size
        self.action_size = action_size  # Set action size
        self.batch_size = batch_size  # Set batch size
        self.memory_size = memory_size  # Set experience replay memory size
        self.memory = deque(maxlen=self.memory_size)  # Create experience replay memory
        self.train_start = train_start  # Set start training step
        self.learning_rate = learning_rate  # Set learning rate
        self.discount_factor = discount_factor  # Set discount factor
        self.epsilon = epsilon  # Set exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate  # Set exploration rate decay rate
        self.ddqn = ddqn_flag  # Set whether to use double deep Q network
        self.dueling_option = dueling_option  # Set Dueling architecture type
        self.polyak_avg = polyak_avg  # Set whether to use Polyak averaging
        self.pa_tau = pa_tau  # Set Polyak averaging weight

        self.model = self._build_model()  # Build the main model
        self.target_model = self._build_model()  # Build the target model

        if load_weights_path is not None:  # If there is a path to load weights
            self.model.load_weights(load_weights_path)  # Load weights
            print('Weights loaded from File')  # Print message

        if load_exp_path is not None:  # If there is a path to load experience
            with open(load_exp_path, 'rb') as file:  # Open file
                self.memory = pickle.load(file)  # Load experience
                print('Experience loaded from file')  # Print message

        self.target_model.set_weights(self.model.get_weights())  # Initialize target model weights to be the same as the main model

    def _build_model(self):  # Define method to build the model
        # Advantage network
        network_input = Input(shape=(self.state_size,))  # Define network input
        A1 = Dense(24, activation='relu')(network_input)  # First fully connected layer
        A2 = Dense(24, activation='relu')(A1)  # Second fully connected layer
        A3 = Dense(self.action_size, activation='linear')(A2)  # Output layer

        # Value network
        V1 = Dense(24, activation='relu')(network_input)  # First fully connected layer
        V2 = Dense(24, activation='relu')(V1)  # Second fully connected layer
        V3 = Dense(1, activation='linear')(V2)  # Output layer

        if self.dueling_option == 'avg':  # If using average Dueling architecture
            network_output = Lambda(lambda x: x[0] - tf.reduce_mean(x[0]) + x[1], output_shape=(self.action_size,))([A3, V3])  # Calculate Q values
        elif self.dueling_option == 'max':  # If using max Dueling architecture
            network_output = Lambda(lambda x: x[0] - tf.reduce_max(x[0]) + x[1], output_shape=(self.action_size,))([A3, V3])  # Calculate Q values
        elif self.dueling_option == 'naive':  # If using naive Dueling architecture
            network_output = Lambda(lambda x: x[0] + x[1], output_shape=(self.action_size,))([A3, V3])  # Calculate Q values
        else:  # If Dueling architecture type is invalid
            raise Exception('Invalid Dueling Option')  # Raise exception

        model = Model(network_input, network_output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def update_target_network(self):
        if self.ddqn and self.polyak_avg:  # If using double deep Q network and Polyak averaging
            weights = self.model.get_weights()  # Get main model weights
            target_weights = self.target_model.get_weights()  # Get target model weights
            for i in range(len(target_weights)):  # Iterate over weights
                target_weights[i] = weights[i] * self.pa_tau + target_weights[i] * (1 - self.pa_tau)  # Update target model weights
            self.target_model.set_weights(target_weights)  # Set target model weights
        else:  # If not using double deep Q network or Polyak averaging
            self.target_model.set_weights(self.model.get_weights())  # Set target model weights to be the same as the main model

    def update_epsilon(self):  # Define method to update exploration rate
        if self.epsilon > 0.01:  # If exploration rate is greater than 0.01
            self.epsilon *= self.epsilon_decay_rate  # Update exploration rate
        else:  # If exploration rate is less than or equal to 0.01
            self.epsilon = 0.01  # Set exploration rate to 0.01

    def add_experience(self, state, action, reward, next_state, done):  # Define method to add experience
        self.memory.append([state, action, reward, next_state, done])  # Append experience to experience replay memory

    def select_action(self, state):  # Define method to select action
        if random.random() < self.epsilon:  # Choose action based on exploration rate
            return random.randint(0, self.action_size - 1)  # Select random action
        else:  # Otherwise
            return np.argmax(self.model.predict(state)[0])  # Select action based on model prediction

    def get_maxQvalue_nextstate(self, next_state):  # Define method to get maximum Q value of next state
        if self.ddqn:  # If using double deep Q network
            action = np.argmax(self.model.predict(next_state)[0])  # Get action chosen by main model in next state
            max_q_value = self.target_model.predict(next_state)[0][action]  # Get Q value of action chosen by target model in next state
        else:  # If not using double deep Q network
            max_q_value = np.amax(self.target_model.predict(next_state)[0])  # Get maximum Q value chosen by target model in next state
        return max_q_value  # Return maximum Q value

    def train_model(self):  # Define method to train model
        if len(self.memory) < self.train_start:  # If number of experiences in experience replay memory is less than start training step
            return  # Do not train

        mini_batch = random.sample(self.memory, self.batch_size)  # Sample a batch of experiences from experience replay memory
        current_state = np.zeros((self.batch_size, self.state_size))  # Create current state array
        next_state = np.zeros((self.batch_size, self.state_size))  # Create next state array
        qValues = np.zeros((self.batch_size, self.action_size))  # Create Q values array

        for i in range(self.batch_size):  # Iterate over batch size
            current_state[i] = mini_batch[i][0]  # Get current state
            action = mini_batch[i][1]  # Get action
            reward = mini_batch[i][2]  # Get reward
            next_state[i] = mini_batch[i][3]  # Get next state
            done = mini_batch[i][4]  # Get done flag

            qValues[i] = self.model.predict(current_state[i].reshape(1, self.state_size))[0]  # Get Q values for current state
            max_qvalue_ns = self.get_maxQvalue_nextstate(next_state[i].reshape(1, self.state_size))  # Get maximum Q value for next state

            if done:
                qValues[i][action] = reward  # Update Q value to reward
            else:
                qValues[i][action] = reward + self.discount_factor * max_qvalue_ns  # Update Q value to reward plus discounted maximum Q value of next state

        self.model.fit(current_state, qValues, batch_size=self.batch_size, epochs=1, verbose=0)  # Train model
        self.update_epsilon()  # Update exploration rate

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # Get state size
action_size = env.action_space.n  # Get action size

deepQ = DQNAgent(state_size, action_size)  # Create DQNAgent object

max_episodes = 1000  # Set maximum episodes
targetNetworkUpdateFreq = 10  # Set target network update frequency
train_start = 1000  # Set start training step
model_save_freq = 300  # Set model save frequency

last100Scores = deque(maxlen=100)  # Create deque for last 100 scores
Scores = []  # Create scores list
AvgScores = []  # Create average scores list
Avg100Scores = []  # Create last 100 average scores list

for e in range(max_episodes):  # Loop over episodes
    state = env.reset()  # Reset environment
    state_array = state[0]  # Get state array
    state_list = state_array.tolist()  # Convert state array to list
    state = np.array(state_list).flatten()  # Flatten state array
    state = np.reshape(state, [1, state_size])  # Reshape state
    t = 0  # Initialize steps
    done = False  # Initialize done flag
    while not done:  # Loop until done
        action = deepQ.select_action(state)  # Select action
        next_state, reward, done, _, _ = env.step(action)  # Execute action
        next_state = np.reshape(next_state, [1, state_size])  # Reshape next state

        reward = reward if not done else -100  # Update reward

        t += 1  # Update steps

        deepQ.add_experience(state, action, reward, next_state, done)  # Add experience
        deepQ.train_model()  # Train model

        state = next_state  # Update state

        if done:
            deepQ.update_target_network()  # Update target network

            Scores.append(t)  # Record score
            AvgScores.append(np.mean(Scores))  # Record average score
            last100Scores.append(t)  # Record last 100 scores
            Avg100Scores.append(np.mean(last100Scores))  # Record last 100 average scores

            print('Episode: {}, time steps: {}, AvgScore: {:0.2f}, epsilon: {:0.2f}, replay size: {}'.format(e, t, np.mean(Scores), deepQ.epsilon, len(deepQ.memory)))

    if np.mean(last100Scores) > (env.spec.max_episode_steps - 5):  # If last 100 average scores is greater than max episodes minus 5
        print('The problem is solved in {} episodes. Exiting'.format(e))  # Print message and exit loop
        break

env.close()  # Close environment

# Plot epsilon-episode graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(deepQ.epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay')

# Plot time steps per episode graph
plt.subplot(1, 3, 2)
plt.plot(Scores)
plt.xlabel('Episode')
plt.ylabel('Time Steps')
plt.title('Time Steps per Episode')

# Plot average score per episode graph
plt.subplot(1, 3, 3)
plt.plot(AvgScores)
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Average Score per Episode')

plt.tight_layout()
plt.show()



