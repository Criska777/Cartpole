import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class CartPoleSolver:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.no_buckets = (1, 1, 6, 3)
        self.no_actions = self.env.action_space.n
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.state_bounds[1] = (-0.5, 0.5)
        self.state_bounds[3] = (-math.radians(50), math.radians(50))
        self.q_table = np.zeros(self.no_buckets + (self.no_actions,))

        # Hyperparameters
        self.min_explore_rate = 0.1
        self.min_learning_rate = 0.1
        self.max_episodes = 1000
        self.max_time_steps = 250
        self.streak_to_end = 120
        self.solved_time = 199
        self.discount = 0.99

        self.no_streaks = 0

        # Tracking metrics
        self.rewards_per_episode = []
        self.times_per_episode = []
        self.avg_times_per_episode = []
        self.learning_rates_per_episode = []
        self.explore_rates_per_episode = []
        self.avg_rewards_per_episode = []  # Added for average score tracking

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_explore_rate(self, episode):
        return max(self.min_explore_rate, min(1.0, 1.0 - math.log10((episode + 1) / 25)))

    def update_learning_rate(self, episode):
        return max(self.min_learning_rate, min(1.0, 1.0 - math.log10((episode + 1) / 25)))

    def bucketize_state(self, state):
        bucket_indices = []

        for i, val in enumerate(state):
            low, high = self.state_bounds[i]
            if val <= low:
                bucket_index = 0
            elif val >= high:
                bucket_index = self.no_buckets[i] - 1
            else:
                scale = (self.no_buckets[i] - 1) / (high - low)
                bucket_index = int(round(scale * (val - low)))
            bucket_indices.append(bucket_index)
        return tuple(bucket_indices)

    def run_episode(self, episode_no):
        explore_rate = self.update_explore_rate(episode_no)
        learning_rate = self.update_learning_rate(episode_no)

        self.learning_rates_per_episode.append(learning_rate)
        self.explore_rates_per_episode.append(explore_rate)

        observation = self.env.reset()
        observation_array = observation[0]
        observation_list = observation_array.tolist()
        state = self.bucketize_state(observation_list)

        done = False
        time_step = 0
        total_reward = 0  # Added for tracking total reward in episode

        while not done:
            action = self.select_action(state, explore_rate)
            next_observation, reward, done, info, _ = self.env.step(action)
            next_state = self.bucketize_state(next_observation)

            best_next_q = np.max(self.q_table[next_state])
            self.q_table[state + (action,)] += learning_rate * (
                        reward + self.discount * best_next_q - self.q_table[state + (action,)])
            state = next_state
            time_step += 1
            total_reward += reward  # Update total reward

            if time_step >= self.solved_time:
                self.no_streaks += 1
            else:
                self.no_streaks = 0

        self.times_per_episode.append(time_step)
        self.rewards_per_episode.append(total_reward)  # Update with total_reward
        self.avg_rewards_per_episode.append(np.mean(self.rewards_per_episode[-100:]))  # Compute average score

        if episode_no % 100 == 0:
            print(f'Episode {episode_no} finished after {time_step} time steps')

    def train(self):
        total_time = 0
        for episode_no in range(self.max_episodes):
            self.run_episode(episode_no)
            total_time += self.times_per_episode[-1]
            self.avg_times_per_episode.append(total_time / (episode_no + 1))

            if self.no_streaks > self.streak_to_end:
                print(f'CartPole problem is solved after {episode_no} episodes.')
                break

        self.env.close()
        self.plot_results()

    def plot_results(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))

        axes[0].plot(self.learning_rates_per_episode, label='Learning Rate')
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel('Learning Rate')
        axes[0].set_title('Learning Rate per Episode')

        axes[1].plot(self.times_per_episode, label='Time per Episode')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel('Time per Episode')
        axes[1].set_title('Time per Episode')

        axes[2].plot(self.avg_rewards_per_episode, label='Average Score')
        axes[2].set_xlabel('Episodes')
        axes[2].set_ylabel('Average Score')
        axes[2].set_title('Average Score per Episode')

        for ax in axes:
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    solver = CartPoleSolver()
    solver.train()
