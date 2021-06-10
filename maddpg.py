from numpy.lib.function_base import select
import torch
import torch.nn.functional as F

from ddpg_agent import Agent, SEED
from experience import ReplayBuffer

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99

class Maddpg():
    def __init__(self, state_size, action_size, num_agents):
        super(Maddpg, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.agents = [Agent(num_agents, state_size, action_size) for _ in range(num_agents)]
        self.memory = ReplayBuffer(BUFFER_SIZE, num_agents, state_size, action_size, SEED)
        self.t_step = 0
        self.update_every = 1

    def act(self, state):
        return [a.act(s) for a, s in zip(self.agents, state)]

    def actions_target(self, states):
        with torch.no_grad():
            actions = torch.empty((BATCH_SIZE, self.num_agents, self.action_size))
            for i, a in enumerate(self.agents):
                actions[:,i] = a.actor_target(states[:,i])
        return actions

    def actions_local(self, states, agent_id):
        actions = torch.empty((BATCH_SIZE, self.num_agents, self.action_size))
        for i, a in enumerate(self.agents):
            action = a.actor(states[:,i])
            actions[:,i] = action if i==agent_id else action.detach()
        return actions

    def q_value(self, agent_id, next_states, rewards):
        view = next_states.view(BATCH_SIZE, -1)
        rewards = rewards[:,agent_id].unsqueeze_(1)
        with torch.no_grad():
            next_actions = self.actions_target(next_states)
            next_actions = next_actions.view(BATCH_SIZE, -1)
            next_q_val = self.agents[agent_id].critic_target(view, next_actions)
            return rewards + GAMMA * next_q_val 

    def store(self, state, actions, rewards, next_state):
        self.memory.store(state, actions, rewards, next_state)

        self.t_step = (self.t_step + 1) % self.update_every
        if len(self.memory) >= BATCH_SIZE and self.t_step == 0:
            self.learn()

    def learn(self):
        for agent_id, agent in enumerate(self.agents):
            states, actions, rewards, next_states = self.memory.sample(BATCH_SIZE)

            actions = actions.view(BATCH_SIZE, -1)
            states_view = states.view(BATCH_SIZE, -1)
            
            # Critic training
            q_value_expected = self.q_value(agent_id, next_states, rewards)
            agent.critic_optimizer.zero_grad()
            states_view = states.view(BATCH_SIZE, -1)
            q_value_predicted = agent.critic(states_view, actions)
            loss = F.mse_loss(q_value_predicted, q_value_expected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
            agent.critic_optimizer.step()

            # Actor training
            agent.actor_optimizer.zero_grad()
            actions_local = self.actions_local(states, agent_id)
            actions_local = actions_local.view(BATCH_SIZE, -1)
            q_value_predicted = agent.critic(states_view, actions_local)
            loss = -q_value_predicted.mean()
            loss.backward()
            agent.actor_optimizer.step()

            # Learning-Rate Update
            agent.lr_step()

        for agent in self.agents:
            agent.soft_update(agent.actor, agent.actor_target)
            agent.soft_update(agent.critic, agent.critic_target)

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()

    def state_dict(self):
        return [agent.actor.state_dict() for agent in self.agents]

    def load_state_dict(self, state_dicts):
        for agent, state_dict in zip(self.agents, state_dicts):
            agent.actor.load_state_dict(state_dict)