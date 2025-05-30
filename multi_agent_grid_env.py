import numpy as np
import random

class MultiAgentGridEnv:
    def __init__(self, grid_size=5, num_agents=2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (0, 0)    # stay
        }
        self.reset()

    def reset(self):
        self.agent_positions = []
        self.delivery_positions = []
        self.done_agents = [False for _ in range(self.num_agents)]
        self.total_steps = 0

        occupied = set()
        for _ in range(self.num_agents):
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in occupied:
                    self.agent_positions.append(pos)
                    occupied.add(pos)
                    break

        for _ in range(self.num_agents):
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in occupied:
                    self.delivery_positions.append(pos)
                    occupied.add(pos)
                    break

        return self._get_obs()

    def _get_obs(self):
        return [self._get_agent_obs(i) for i in range(self.num_agents)]

    def _get_agent_obs(self, i):
        return np.array(self.agent_positions[i] + self.delivery_positions[i])

    def _in_bounds(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def step(self, actions):
        rewards = [0 for _ in range(self.num_agents)]
        new_positions = []
        collisions = set()

        for i, action in enumerate(actions):
            if self.done_agents[i]:
                new_positions.append(self.agent_positions[i])
                continue

            move = self.action_map[action]
            new_pos = (self.agent_positions[i][0] + move[0],
                       self.agent_positions[i][1] + move[1])

            if not self._in_bounds(new_pos):
                rewards[i] -= 10
                new_positions.append(self.agent_positions[i])
                continue

            new_positions.append(new_pos)

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if new_positions[i] == new_positions[j] and not self.done_agents[i] and not self.done_agents[j]:
                    collisions.add(i)
                    collisions.add(j)

        for i in range(self.num_agents):
            if self.done_agents[i]:
                continue

            if i in collisions:
                rewards[i] -= 5
                new_positions[i] = self.agent_positions[i]
            else:
                self.agent_positions[i] = new_positions[i]
                rewards[i] -= 1

                if self.agent_positions[i] == self.delivery_positions[i]:
                    rewards[i] += 10
                    if i not in collisions:
                        rewards[i] += 5
                    self.done_agents[i] = True

        done = all(self.done_agents)
        self.total_steps += 1
        if self.total_steps >= 100:
            done = True

        return self._get_obs(), rewards, done, {}
