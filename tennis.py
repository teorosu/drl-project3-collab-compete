# %% [markdown]
# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# %%
from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

from maddpg import Maddpg

# %% [markdown]
# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Tennis.app"`
# - **Windows** (x86): `"path/to/Tennis_Windows_x86/Tennis.exe"`
# - **Windows** (x86_64): `"path/to/Tennis_Windows_x86_64/Tennis.exe"`
# - **Linux** (x86): `"path/to/Tennis_Linux/Tennis.x86"`
# - **Linux** (x86_64): `"path/to/Tennis_Linux/Tennis.x86_64"`
# - **Linux** (x86, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86"`
# - **Linux** (x86_64, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Tennis.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Tennis.app")
# ```

# %%
env = UnityEnvironment(file_name="Tennis.app")

# %%
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# %% [markdown]
# ### Examine the State and Action Spaces
# 
# In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# 
# The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
# 
# Run the code cell below to print some information about the environment.

# %%
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
state = env_info.vector_observations
state_size = state.shape[1]

print('There are {} agents. Each observes a state with length: {}'
      .format(state.shape[0], state_size))
print('The observation for the first agent looks like:\n', state[0])

# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```

# Instantiate a Multi Agent
maddpg = Maddpg(state_size, action_size, num_agents)

# %%
## Define the training function
def train(env, maddpg, max_episodes=2000):
    moving_window = deque(maxlen=100)
    scores_list = []  # list containing scores from each episode
    
    ## Perform n_episodes of training
    for i_episode in range(1, max_episodes+1):
        scores = []
        maddpg.reset_noise()
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        scores_episode = np.zeros(num_agents)
        while True:
            actions = maddpg.act(state)
            env_info = env.step(actions)[brain_name]

            rewards = env_info.rewards
            next_state = env_info.vector_observations

            maddpg.store(state, actions, rewards, next_state)

            state = next_state
            scores_episode += rewards

            if any(env_info.local_done):
                break

        scores.append(scores_episode.max())
        scores_list.append(scores_episode.max())
        moving_window.append(np.mean(scores))

        print('\rEpisode {:4d}\t Last score: {:5.2f} ({:5.2f} / {:5.2f})\tMoving average: {:5.3f}'
              .format(i_episode, scores[-1], scores_episode[0], scores_episode[1], np.mean(moving_window)), end='')
        if i_episode % 100 == 0:
            print()

        if np.mean(moving_window) >= 0.5 and i_episode >= 100:
            print('\n\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'
                  .format(i_episode-100, np.mean(moving_window)))
            torch.save(maddpg.state_dict(), 'checkpoint.pth')
            break

    return scores_list


# %%
## Train the agent
scores_list = train(env, maddpg)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_list)+1), scores_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# %%
## Test the trained model
def test(env, maddpg, max_episodes=3):
    """Test a Multi Agent Deep Deterministic Policy Gradients (MADDPG).
    Params
    ======
        env (UnityEnvironment): environment for the agents
        maddpg (MADDPG): the Multi Agent DDPG
        n_episodes (int): maximum number of training episodes
    """
    ## Perform n_episodes of training
    for i_episode in range(1, max_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        scores = np.zeros(num_agents)
        while True:
            states = env_info.vector_observations
            actions = maddpg.act(states)
            env_info = env.step(actions)[brain_name]
            scores += env_info.rewards
            states = env_info.vector_observations
            if any(env_info.local_done):
                break
        print('\rEpisode {:4d}\tScore: {:5.2f} ({:5.2f} / {:5.2f})\t'
              .format(i_episode, scores.max(), scores[0], scores[1]))

maddpg = Maddpg(state_size, action_size, num_agents)
maddpg.load_state_dict(torch.load('checkpoint.pth'))
test(env, maddpg)


# %%
env.close()
