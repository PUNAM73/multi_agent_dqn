import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym

from dqn_agent import DQNAgent

print("Torch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Gym version:", gym.__version__)

# Test action selection
state_size = 3
action_size = 5
agent = DQNAgent(state_size=state_size, action_size=action_size)

sample_state = np.random.rand(state_size)
action = agent.act(sample_state)
print("Sample action:", action)