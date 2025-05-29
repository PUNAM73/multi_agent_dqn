from grid_env import GridEnv
env = GridEnv()
state = env.reset()
print("Initial state:", state)
for _ in range(10):
    action = int(input("Action (0=up, 1=down, 2=left, 3=right, 4=stay): "))
    next_state, reward, done = env.step(action)
    print("Next state:", next_state, "| Reward:", reward, "| Done:", done)
    if done:
        break
