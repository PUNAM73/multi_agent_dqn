import torch
import numpy as np
from dqn_agent import DQNAgent
from grid_env import GridEnv

# Hyperparameters
num_episodes = 1500
max_steps_per_episode = 100
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
gamma = 0.99
learning_rate = 0.001
batch_size = 64
target_update_freq = 10

# Initialize environment and agent
env = GridEnv(grid_size=5)
state_size = env.get_state().shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size, gamma, learning_rate, batch_size, seed=42)


# Stats tracking
total_steps = 0
total_deliveries = 0
collision_free_deliveries = 0  # no collision in single agent
total_cost = 0

# Training loop
epsilon = epsilon_start

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        total_steps += 1

        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)

        total_cost += -reward if reward < 0 else 0  # accumulate cost
        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        episode_reward += reward

        if done:
            if info.get("success", False):
                total_deliveries += 1
                collision_free_deliveries += 1
            break

    if episode % target_update_freq == 0:
        agent.update_target_network()

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Progress print
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {epsilon:.2f}")

# Final Stats
print("\nFinal Stats:")
print(f"Total Episodes: {episode + 1}")
print(f"Total Steps: {total_steps}")
print(f"Total Deliveries: {total_deliveries}")
print(f"Collision-Free Deliveries: {collision_free_deliveries}")
print(f"Total Cost (negative reward): {total_cost:.2f}")
print(f"Collision-Free Delivery Rate: {collision_free_deliveries / (episode + 1):.2f}")
