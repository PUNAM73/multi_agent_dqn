import numpy as np

class MultiAgentGridEnv:
    def __init__(self, grid_size=5, num_agents=2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.pickup_locations = [(0, 0), (0, 2), (4, 0), (2, 4)]
        self.dropoff_location = (4, 4)
        self.reset()

    def reset(self):
        self.agent_positions = []
        self.item_picked = [False] * self.num_agents
        self.delivered = [False] * self.num_agents

        occupied = set()
        for _ in range(self.num_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
                if pos not in occupied and pos not in self.pickup_locations and pos != self.dropoff_location:
                    self.agent_positions.append(pos)
                    occupied.add(pos)
                    break
        return self.get_state()

    def get_state(self):
        return [np.array([pos[0], pos[1], int(pick)]) for pos, pick in zip(self.agent_positions, self.item_picked)]

    def step(self, actions):
        rewards = [0 for _ in range(self.num_agents)]
        new_positions = list(self.agent_positions)
        infos = [{} for _ in range(self.num_agents)]

        for i, action in enumerate(actions):
            r, c = self.agent_positions[i]
            if action == 0 and r > 0: r -= 1
            elif action == 1 and r < self.grid_size - 1: r += 1
            elif action == 2 and c > 0: c -= 1
            elif action == 3 and c < self.grid_size - 1: c += 1
            new_positions[i] = (r, c)

        if len(set(new_positions)) < len(new_positions):
            for i in range(self.num_agents):
                if new_positions.count(new_positions[i]) > 1:
                    new_positions[i] = self.agent_positions[i]
                    rewards[i] -= 10

        self.agent_positions = new_positions

        for i in range(self.num_agents):
            pos = self.agent_positions[i]
            pickup_location = self.pickup_locations[i % len(self.pickup_locations)]

            if not self.item_picked[i] and pos == pickup_location:
                self.item_picked[i] = True
                rewards[i] += 10

            elif self.item_picked[i] and pos == self.dropoff_location:
                self.item_picked[i] = False
                self.delivered[i] = True
                rewards[i] += 20
                infos[i]["delivery"] = "succeeded"

            elif not self.item_picked[i] and pos == self.dropoff_location:
                rewards[i] -= 5
                infos[i]["delivery"] = "failed"

        dones = [self.delivered[i] for i in range(self.num_agents)]
        return self.get_state(), rewards, dones, infos
