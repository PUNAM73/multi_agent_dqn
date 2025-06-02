# # import numpy as np
# import torch
# from multi_agent_grid_env import MultiAgentGridEnv
# from dqn_agent import DQNAgent

# NUM_EPISODES = 1000
# MAX_STEPS = 100
# GAMMA = 0.99
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 0.995
# LR = 1e-4
# BATCH_SIZE = 64
# TARGET_UPDATE_FREQ = 100

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# env = MultiAgentGridEnv(grid_size=5, num_agents=2)
# num_agents = env.num_agents
# obs_dim = len(env._get_obs()[0])
# n_actions = 5

# agents = [
#     DQNAgent(
#         obs_dim,
#         n_actions,
#         lr=LR,
#         gamma=GAMMA,
#         epsilon=EPSILON_START,
#         epsilon_min=EPSILON_END,
#         epsilon_decay=EPSILON_DECAY,
#         batch_size=BATCH_SIZE,
#         target_update_freq=TARGET_UPDATE_FREQ,
#         device=device
#     )
#     for _ in range(num_agents)
# ]

# total_steps_all_episodes = []

# total_success = 0
# collision_free_success = 0
# max_collisions_allowed = 4
# step_budget = 1500

# total_collisions = 0
# total_steps = 0

# epsilon = EPSILON_START  # Initialize epsilon before training

# for ep in range(NUM_EPISODES):
#     obs = env.reset()
#     done = False
#     episode_rewards = [0] * num_agents
#     steps = 0
#     delivery_flags = [False] * num_agents
#     collision_flags = [False] * num_agents

#     while not done and steps < MAX_STEPS:
#         actions = [agents[i].act(obs[i], eps=epsilon) for i in range(num_agents)]
#         next_obs, rewards, done, _ = env.step(actions)

#         for i in range(num_agents):
#             if rewards[i] < -4:
#                 collision_flags[i] = True
#             if rewards[i] >= 10 and not delivery_flags[i]:
#                 delivery_flags[i] = True

#         for i in range(num_agents):
#             agents[i].step(obs[i], actions[i], rewards[i], next_obs[i], done)

#         obs = next_obs
#         steps += 1
#         for i in range(num_agents):
#             episode_rewards[i] += rewards[i]

#     total_steps += steps
#     success = sum(delivery_flags)
#     collision_free = sum(1 for i in range(num_agents) if delivery_flags[i] and not collision_flags[i])
#     collisions = sum(collision_flags)

#     total_success += success
#     collision_free_success += collision_free
#     total_collisions += collisions

#     print(f"Episode {ep+1}: Steps={steps}, Deliveries={success}, "
#           f"Collision-Free={collision_free}, Collisions={collisions}")

#     # Decay epsilon after each episode
#     epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

#     # Stop training early if criteria met
#     if total_steps >= step_budget and total_collisions <= max_collisions_allowed and (collision_free_success / ((ep+1)*num_agents)) >= 0.75:
#         print("\nTraining goal achieved.")
#         break

# print("\nFinal Stats:")
# print(f"Total Episodes: {ep+1}")
# print(f"Total Steps: {total_steps}")
# print(f"Total Deliveries: {total_success}")
# print(f"Collision-Free Deliveries: {collision_free_success}")
# print(f"Total Collisions: {total_collisions}")
# # print(f"Collision-Free Delivery Rate: {collision_free_success / ((ep+1)*num_agents):.2f}") 


import torch
import numpy as np
from dqn_agent import DQNAgent
from multi_agent_grid_env import MultiAgentGridEnv

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

# Initialize environment and agents
env = MultiAgentGridEnv(grid_size=5, num_agents=2)
state_size = len(env.reset()[0])  # Get the size of one agent's observation
action_size = len(env.action_map)

agents = [
    DQNAgent(state_size, action_size, gamma, learning_rate, batch_size, seed=42),
    DQNAgent(state_size, action_size, gamma, learning_rate, batch_size, seed=43)
]

# Stats tracking
total_steps = 0
total_deliveries = [0, 0]
total_fails = [0, 0]
total_rewards = [0, 0]
total_cost = 0

# Training loop
epsilon = epsilon_start

for episode in range(num_episodes):
    states = env.reset()
    episode_rewards = [0, 0]
    done = False

    for step in range(max_steps_per_episode):
        total_steps += 1

        actions = [agent.act(state, epsilon) for agent, state in zip(agents, states)]
        next_states, rewards, done, _ = env.step(actions)

        for i in range(2):  # 2 agents
            if rewards[i] < 0:
                total_cost += -rewards[i]  # accumulate negative rewards as cost
            agents[i].replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], done)
            agents[i].update()
            episode_rewards[i] += rewards[i]

        states = next_states

        if done:
            for i in range(2):
                if rewards[i] >= 10:  # Successful delivery
                    total_deliveries[i] += 1
                else:
                    total_fails[i] += 1
            break

    if episode % target_update_freq == 0:
        for agent in agents:
            agent.update_target_network()

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Print every 50 episodes
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1:4d}  | Total Reward: {sum(episode_rewards):.2f}  | "
              f"Epsilon: {epsilon:.2f} | Alpha (Ir): {learning_rate:.4f}")
        print(f"Agent 1 Reward: {episode_rewards[0]:.2f} | Deliveries Succeeded: {total_deliveries[0]}  | Failed: {total_fails[0]}")
        print(f"Agent 2 Reward: {episode_rewards[1]:.2f} | Deliveries Succeeded: {total_deliveries[1]}  | Failed: {total_fails[1]}")
        print("-" * 80)

# Final Stats
print("\nFinal Stats:")
print(f"Total Episodes: {num_episodes}")
print(f"Total Steps: {total_steps}")
print(f"Total Deliveries: {sum(total_deliveries)}")
print(f"Collision-Free Deliveries: {sum(total_deliveries)}")
print(f"Total Cost (negative reward): {total_cost:.2f}")
print(f"Collision-Free Delivery Rate: {sum(total_deliveries) / num_episodes:.2f}")
print(f"Alpha (Learning Rate): {learning_rate}")
