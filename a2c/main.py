import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, rpc_sync, remote

from coordinator import Coordinator
import config

import numpy as np
import json
import os

#torchrun --nnodes 1 --nproc_per_node 8 rpc_rl_batchsize.py

'''
Architecture:
rank=0:
    Create Policy
        remote call: get action | state
        remote call: send reward (log prob local)
    Wrap in Coordinator:
        call: spin up K workers, init EpisodeRunner, run episode
        call: once episodes finish, run PG update
rank!=0:
    Create EpisodeRunner
        Init env + run episode + get actions through rpc calls + send rewards through rpc calls
'''


key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

def run(rank, world_size):
    print(f'rank = {rank} world_size = {world_size}')
    env_dict = {key: os.environ[key] for key in key_list}

    print(env_dict)

    if rank==0:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)

        coordinator = Coordinator(world_size, config.batch_size_multiple, lr=config.lr)
        coord_rref = RRef(coordinator)
        coordinator.run_training_loop(config.n_iter, coord_rref)

        torch.save(coordinator.policy, open(f'plots/{coordinator.env_name}_policy_nworkers{world_size-1}_batchsizemultiple{batch_size_multiple}.pt', 'wb'))
        json.dump(coordinator.stats, open(f'plots/{coordinator.env_name}_stats_nworkers{world_size-1}_batchsizemultiple{batch_size_multiple}.json', 'w'))

    else:
        rpc.init_rpc(f"rank{rank}", rank=rank, world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE)

    rpc.shutdown()

if __name__=='__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    run(rank, world_size)
        