import torch
import numpy as np

import envwrapper
import agent

torch.manual_seed(0)
np.random.seed(0)

# Create environment
env = envwrapper.Env(no_graphics=False)
nA, nS = env.return_sizes()


# ACHIEVES ABOUT 15.97 POINTS AFTER ABOUT 2300 ITERATIONS
agent_dict={
    "num_episodes": 3000,
    "name": "rl_agent",
    "save_after": 2300,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01,
    "tau": 0.1,
    "num_replays": 2,
    "memory_size": 2**20,
    "batchsize": 2**11,
    "replay_reg": 0.0,
    "seed": 0
}

model_dict = {
    "input_size": 37,
    "output_size": 4,
    "hn": [64, 32, 16],
    "dueling": True
}

# Create agent 
a = agent.Agent(agent_dict=agent_dict, model_dict = model_dict)

# Train agent
a.run(env)

