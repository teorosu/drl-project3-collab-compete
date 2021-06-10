import random
from collections import deque

import torch

class ReplayBuffer:
    def __init__(self, capacity, num_agents, state_size, action_size, seed=13):
        random.seed(seed)
        # Configs
        self.num_agents = num_agents
        self.capacity = capacity
        self.state_size = state_size
        self.action_size = action_size

        # Storage
        self.memory = deque(maxlen=capacity)

    def store(self, state, actions, reward, next_state):
        self.memory.append((state, actions, reward, next_state))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)

        states = torch.empty((batch_size, self.num_agents, self.state_size), dtype=torch.float)
        next_states = torch.empty((batch_size, self.num_agents, self.state_size), dtype=torch.float)
        actions = torch.empty((batch_size, self.num_agents, self.action_size), dtype=torch.float)
        rewards = torch.empty((batch_size, self.num_agents), dtype=torch.float)

        for i, e in enumerate(experiences):
            states[i], actions[i], rewards[i], next_states[i] = map(torch.as_tensor, e)
        return (states, actions, rewards, next_states)

    def __len__(self):
        return len(self.memory)
