import copy
import random

import numpy as np
import torch
import torch.optim as optim

from model import Actor, Critic

SEED=13
LR_SCHED_STEP=1000
LR_SCHED_GAMMA=0.99
ACTOR_LR=3e-3
CRITIC_LR=4e-4
TAU=8e-3
OU_NOISE_THETA=0.9
OU_NOISE_SIGMA=0.01

class Agent():
    def __init__(self, num_agents, state_size, action_size):
        random.seed(SEED)

        # Configs
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network
        self.actor = Actor(state_size, action_size, fc1_units=128, fc2_units=64, seed=SEED)
        self.actor_target = Actor(state_size, action_size, fc1_units=128, fc2_units=64, seed=SEED)
        self.soft_update(self.actor, self.actor_target, 1)

        # Critic Network
        self.critic = Critic(state_size, action_size, num_agents, fc1_units=128, fc2_units=64, seed=SEED)
        self.critic_target = Critic(state_size, action_size, num_agents, fc1_units=128, fc2_units=64, seed=SEED)
        self.soft_update(self.critic, self.critic_target, 1)

        # Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.actor_lr_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=LR_SCHED_STEP, gamma=LR_SCHED_GAMMA)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.critic_lr_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=LR_SCHED_STEP, gamma=LR_SCHED_GAMMA)

        # Initialize a noise process
        self.noise = OUNoise(action_size)

    def soft_update(self, local_model, target_model, tau=TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def act(self, state):
        with torch.no_grad():
            self.actor.eval()
            state = torch.from_numpy(state).float()
            action = self.actor(state).data.cpu().numpy()
            self.actor.train()

        action += self.noise.sample()
        np.clip(action, a_min=-1, a_max=1, out=action)

        return action

    def lr_step(self):
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

    def reset_noise(self):
        self.noise.reset()

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, action_size, mu=0.):
        """Initialize parameters and noise process."""
        random.seed(SEED)
        self.mu = mu * np.ones(action_size)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        random_array = [random.random() for i in range(len(x))]
        dx = OU_NOISE_THETA * (self.mu - x) + OU_NOISE_SIGMA * np.array(random_array)
        self.state = x + dx
        return self.state
