import numpy as np
import gym

import torch.nn as nn

batch_size_multiple = 5
n_iter = 1000
lr = 1e-3

env_name = 'LunarLander-v2'
env = gym.make(env_name)

if not isinstance(env.action_space, gym.spaces.Discrete):
    raise ValueError(f"env = {env_name} doesn't have Discrete action_space {env.action_space}")

N_inputs = np.prod(env.observation_space.shape)
N_outputs = env.action_space.n

N_hidden_layers = 1
N_hidden_nodes = 10
activation = nn.ReLU()
output_activation = nn.Softmax(dim=1)
