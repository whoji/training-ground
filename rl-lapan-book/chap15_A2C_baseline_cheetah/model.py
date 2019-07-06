import torch
import torch.nn as nn
import numpy as np

HID_SIZE = 64

class ActorNet(nn.Module):
    """docstring for ActorNet"""
    def __init__(self, obs_size, act_size):
        super(ActorNet, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh() # output range (-1, 1)
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh()
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh()
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)


class CriticNet(nn.Module):
    """docstring for CriticNet"""
    def __init__(self, obs_size):
        super(CriticNet, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU()
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh()
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)


