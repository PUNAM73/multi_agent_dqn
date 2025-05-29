import torch
from grid_env import GridEnv
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GridEnv(grid_size=5)
state_size = 3
action_size = 5

agent = DQNAgent(state_size, action_size, device=device)

n_episodes = 500
max_steps = 500  # large enough to cover full episode or just remove it and rely on done

eps_start = 1.0
eps_end = 0.1
eps_decay = 0.995
alpha = 1e-3

epsilon = eps_start

for episode in range(1, n_episodes + 1):
    state = env.reset()
    total_reward = 0
    deliveries_succeeded = 0
    deliveries_failed = 0

    for step in range(max_steps):
        action = agent.act(state, epsilon)
        
        # Unpack only 3 values from env.step (change this if your env returns info as well)
        next_state, reward, done = env.step(action)
        
        # If you want to use 'info', define it here as empty dict or modify env.step to return it
        info = {}

        agent.step(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # If your environment doesn't return info with delivery status, this will always skip
        if info.get("delivery") == "succeeded":
            deliveries_succeeded += 1
        elif info.get("delivery") == "failed":
            deliveries_failed += 1

        if done:
            break

    epsilon = max(eps_end, eps_decay * epsilon)

    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f} | Alpha (lr): {alpha:.4f}")
    print(f"Deliveries Succeeded: {deliveries_succeeded} | Deliveries Failed: {deliveries_failed}")
