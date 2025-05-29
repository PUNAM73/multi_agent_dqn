import numpy as np
import torch
from multi_agent_grid_env import MultiAgentGridEnv
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = MultiAgentGridEnv(grid_size=5, num_agents=2)
state_size = 3
action_size = 5
num_agents = env.num_agents

agents = [DQNAgent(state_size, action_size, device=device) for _ in range(num_agents)]

n_episodes = 1000
max_steps = 100
eps_start = 1.0
eps_end = 0.1
eps_decay = 0.995
alpha = 1e-3
print_every = 50

epsilon = eps_start

for episode in range(1, n_episodes + 1):
    states = env.reset()
    total_reward = 0
    deliveries_succeeded = 0
    deliveries_failed = 0

    # To keep track per agent if you want
    per_agent_rewards = [0 for _ in range(num_agents)]
    per_agent_success = [0 for _ in range(num_agents)]
    per_agent_failure = [0 for _ in range(num_agents)]

    for step in range(max_steps):
        actions = [agent.act(states[i], epsilon) for i, agent in enumerate(agents)]
        next_states, rewards, dones, infos = env.step(actions)

        # Train agents
        for i, agent in enumerate(agents):
            agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            per_agent_rewards[i] += rewards[i]

        states = next_states
        total_reward += sum(rewards)

        # Assuming infos contains info dict per agent for deliveries:
        for i, info in enumerate(infos):
            if info.get("delivery") == "succeeded":
                deliveries_succeeded += 1
                per_agent_success[i] += 1
            elif info.get("delivery") == "failed":
                deliveries_failed += 1
                per_agent_failure[i] += 1

        if all(dones):
            break

    epsilon = max(eps_end, eps_decay * epsilon)

    # Print detailed results every episode or every N episodes
    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f} | Alpha (lr): {alpha:.4f}")
    for i in range(num_agents):
        print(f" Agent {i+1} Reward: {per_agent_rewards[i]:.2f} | Deliveries Succeeded: {per_agent_success[i]} | Failed: {per_agent_failure[i]}")

