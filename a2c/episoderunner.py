import torch
import torch.nn as nn

import numpy as np
import gym

class EpisodeRunner:
    def __init__(self, env_name, rank):
        self.env = gym.make(env_name)
        self.action_space = np.arange(0, self.env.action_space.n)
        self.rank = rank

    def run_episode(self, coord_rref, batch_id):
        state = self.env.reset()

        done = False

        while not done:
            action_selected_index = coord_rref.rpc_sync().get_action(state, self.rank, batch_id) #rpc call to receive action from worker = 0

            action = self.action_space[action_selected_index]

            state, reward, done, info = self.env.step(action)

            coord_rref.rpc_sync().record_reward(reward, self.rank, batch_id) #rpc call to send reward to worker = 0
