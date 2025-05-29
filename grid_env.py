import numpy as np

class GridEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.pickup_location = (0, 0)
        self.dropoff_location = (4, 4)
        self.reset()

    def reset(self):
        self.agent_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        self.item_picked = False
        self.delivered = False
        return self.get_state()

    def get_state(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], int(self.item_picked)])

    def step(self, action):
        r, c = self.agent_pos
        if action == 0 and r > 0: r -= 1
        elif action == 1 and r < self.grid_size - 1: r += 1
        elif action == 2 and c > 0: c -= 1
        elif action == 3 and c < self.grid_size - 1: c += 1

        self.agent_pos = (r, c)
        reward = 0

        if not self.item_picked and self.agent_pos == self.pickup_location:
            self.item_picked = True
            reward += 10

        elif self.item_picked and self.agent_pos == self.dropoff_location:
            self.item_picked = False
            self.delivered = True
            reward += 20

        elif not self.item_picked and self.agent_pos == self.dropoff_location:
            reward -= 5

        done = self.delivered
        return self.get_state(), reward, done
