import torch
import envwrapper
import agent
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


env = envwrapper.Env(no_graphics=False)
nA, nS = env.return_sizes()

agent_dict={
    "name": "rl_agent",
}

a = agent.Agent(agent_dict=agent_dict)

a.load_state()

a.evaluate(env, delay=0.0)
