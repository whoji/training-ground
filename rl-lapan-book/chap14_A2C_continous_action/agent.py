import torch
import torch.nn as nn
import numpy as np

# this is taken from ptan's agent class
# https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class AgentA2C():
    def __init__(self, net, device='cpu'):
        self.nt = net
        self.device = device

    def predict(self, states, agent_states):
        s_v = float32_preprocessor(states).to(self.device)
        mu_v, var_v, _ = self.net(s_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states

