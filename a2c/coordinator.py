import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.rpc as rpc

import numpy as np
import itertools
import time

from model import PolicyNet
from episoderunner import EpisodeRunner
import config

class Coordinator():
    def __init__(self, world_size, batch_size_multiple, lr=1e-2, gamma=0.99):
        self.env_name = config.env_name
        N_inputs = config.N_inputs
        N_outputs = config.N_outputs

        N_hidden_layers = config.N_hidden_layers
        N_hidden_nodes = config.N_hidden_nodes
        activation = config.activation
        output_activation = config.output_activation

        self.world_size = world_size
        self.batch_size_multiple = batch_size_multiple

        self.policy = PolicyNet(N_inputs, 
                                N_outputs, 
                                N_hidden_layers, 
                                N_hidden_nodes,
                                activation,
                                output_activation=output_activation)        
    

        #data structures to hold rewards and log probs
        #map worker id -> [] of log probs and rewards
        self.log_probs = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}
        self.rewards = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}

        #updating model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        #miscellaneous
        self.rref_list = []
        self.iter = 0
        self.log_interval = 10
        self.stats = {}
        self.start_time = time.time()
        self.exp_reward_factor = 0.90
        self.exp_reward_avg = 0

    def get_action(self, state, worker_id, batch_id):
        #predict prob distribution on actions and sample
        action_probs = self.policy(torch.tensor(state).float().unsqueeze(0))[0]
        action_selected_index = torch.multinomial(action_probs, 1).item()
        
        #update log probs needed for policy gradient updates
        self.log_probs[(worker_id, batch_id)].append(action_probs[[action_selected_index]].log())

        return action_selected_index

    def record_reward(self, reward, worker_id, batch_id):
        self.rewards[(worker_id, batch_id)].append(reward)

        #assert(len(self.rewards[(worker_id, batch_id)]==self.log_probs[(worker_id, batch_id)]))

    def update_model(self):
        #compute J = expected reward
        J = 0
        total_rewards_list = []
        '''
        for k in range(1, self.world_size):
            total_log_prob = torch.cat(self.log_probs[k]).sum()
            total_reward = np.sum(self.rewards[k])

            J += (total_log_prob)*(total_reward)

            total_rewards_list.append(total_reward)
        '''

        '''
        TODO:
        1. Simple baseline
            Consider per-trajectory rewards, R_k (discounting/no discounting, to-go/full)
            Subtract exponential moving averages (computed from per-trajectory rewards above)

        2. Train value-function:
            For each batch, collect tuples: (s, reward-to-go from s)
            Use previous network for current update
            Update V with new data: how many iterations? MSE criterion?

        3. Full actor-critic: replace MC estimate with Q-function

        '''

        for k in self.rewards.keys(): #loop over trajectories
            r = np.array(self.rewards[k])
            episode_length = len(r)

            r_to_go = torch.tensor([np.sum(r[t:] * self.gamma**(np.arange(episode_length-t))) for t in range(episode_length)]) #\Sigma_{t to T} \gamme^(t'-t) r(t')

            J += (torch.cat(self.log_probs[k]) * r_to_go).sum()

            total_reward = np.sum(self.rewards[k])
            total_rewards_list.append(total_reward)

        #baseline is just MC average of current episode
        self.current_reward_avg = np.mean(total_rewards_list)
        if self.iter == 0:
            self.exp_reward_avg = self.current_reward_avg
        else:
            self.exp_reward_avg = self.exp_reward_factor*self.exp_reward_avg + (1-self.exp_reward_factor)*self.current_reward_avg

        #backprop and update policy
        J /= len(self.rewards)
        self.optimizer.zero_grad()
        (-J).backward()
        self.optimizer.step()

        if self.iter % self.log_interval == 0:
            print(f"Current Average Reward: {self.current_reward_avg} Exponentially Average Reward: {self.exp_reward_avg}")

        self.log_probs = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}
        self.rewards = {(i,j):[] for i,j in itertools.product(range(1, self.world_size), range(self.batch_size_multiple))}
        
        self.stats[time.time()-self.start_time] = (self.current_reward_avg, self.exp_reward_avg)
        self.iter += 1
        
    def run_training_loop(self, N_iter, coord_rref):
        self.iter = 0

        if len(self.rref_list)==0:
            self.rref_list = [rpc.remote(f"rank{j}", EpisodeRunner, (self.env_name, j,)) for j in range(1, self.world_size)]

        for i in range(N_iter):           
            #launch episodes
            for batch_id in range(self.batch_size_multiple):
                fut_list = [r.rpc_async().run_episode(coord_rref, batch_id) for r in self.rref_list]
                [fut.wait() for fut in fut_list]

            #update model
            self.update_model()

            #print stats
