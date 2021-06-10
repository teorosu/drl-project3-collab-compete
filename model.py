import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=2, fc1_units=128, fc2_units=64, seed=13):
        super(Actor, self).__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_params()
        
    def reset_params(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=2, num_agents=2, fc1_units=128, fc2_units=64, seed=13):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size * num_agents, fc1_units)
        self.bn = nn.BatchNorm1d(fc1_units + action_size * num_agents)
        self.fc2 = nn.Linear(fc1_units + action_size * num_agents, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_params()

    def reset_params(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
