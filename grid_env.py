import numpy as np

class GridEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.pickup_location = (0, 0)
        self.dropoff_location = (4, 4)
        self.num_features = 3  # row, col, item_picked
        self.observation_space = np.zeros(self.num_features, dtype=np.float32)  # Dummy shape
        self.action_space = type('ActionSpace', (), {'n': 4})()  # 4 discrete actions: up/down/left/right
        self.reset()

    def reset(self):
        self.agent_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        self.item_picked = False
        self.delivered = False
        self.visited_positions = set()
        self.visited_positions.add(self.agent_pos)
        return self.get_state()

    def get_state(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], int(self.item_picked)], dtype=np.float32)

    def step(self, action):
        r, c = self.agent_pos

        # Move logic
        if action == 0 and r > 0: r -= 1  # Up
        elif action == 1 and r < self.grid_size - 1: r += 1  # Down
        elif action == 2 and c > 0: c -= 1  # Left
        elif action == 3 and c < self.grid_size - 1: c += 1  # Right

        self.agent_pos = (r, c)

        # Cost for taking a step
        reward = -1

        # Reward for picking up item
        if not self.item_picked and self.agent_pos == self.pickup_location:
            self.item_picked = True
            reward += 10

        # Reward for successful delivery
        elif self.item_picked and self.agent_pos == self.dropoff_location:
            self.item_picked = False
            self.delivered = True
            reward += 20

        # Penalty for visiting drop-off without item
        elif not self.item_picked and self.agent_pos == self.dropoff_location:
            reward -= 5

        # Optional: extra penalty for visiting same position again
        if self.agent_pos in self.visited_positions:
            reward -= 0.5  # slight penalty for revisiting
        else:
            self.visited_positions.add(self.agent_pos)

        done = self.delivered
        info = {"success": self.delivered}
        return self.get_state(), reward, done, info
