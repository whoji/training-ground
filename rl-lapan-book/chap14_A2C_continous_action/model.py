import torch
import torch.nn as nn
import numpy as np

HID_SIZE = 128

class A2CModel(nn.Module):
    """docstring for A2CModel"""
    def __init__(self, obs_size, act_size):
        super(A2CModel, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh() # output range (-1, 1)
        )

        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus() # output range (0, inf), relu is [0, inf]; we dont want 0 var.
        )

        self.value = nn.Linear(HID_SIZE, 1) # no activation here for value V func

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)
