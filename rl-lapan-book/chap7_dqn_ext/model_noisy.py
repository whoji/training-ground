import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Linear):
    """docstring for NoisyLinear"""
    def __init__(self, in_features, out_features, signma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__( in_features, out_features, bias=True)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), signma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,),signma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight* self.epsilon_weight, bias)



class DQN(nn.Module):
    """docstring for DQN"""
    def __init__(self, input_shape, n_actions):
        """assumes input_shape is of CHW shape (4,84,84)"""
        super(DQN, self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        temp_conv_out = self.conv(torch.zeros(1, *shape))
        # import pdb; pdb.set_trace()
        return int(np.prod(temp_conv_out.size())) # 1*128*7*7 = 6272

    def forward(self, x):
        """ assumes x is 4D of shape BCHW, output will be 2D: B*n_actions """
        conv_out = self.conv(x).view(x.size()[0], -1) # flatten ? what this is not 1D ??!!
        return self.fc(conv_out)


class DuelingDQN(nn.Module):
    """docstring for DuelingDQN"""
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.fc_v = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

    def _get_conv_out(self, shape):
        temp_conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(temp_conv_out.size()))

    def forward(self,x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(x.size()[0], -1)
        v   = self.fc_v(conv_out)
        adv = self.fc_advantage(conv_out)
        return v + adv - adv.mean()


class NoisyDQN(nn.Module):
    """docstring for DQN"""
    def __init__(self, input_shape, n_actions):
        """assumes input_shape is of CHW shape (4,84,84)"""
        super(NoisyDQN, self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        temp_conv_out = self.conv(torch.zeros(1, *shape))
        # import pdb; pdb.set_trace()
        return int(np.prod(temp_conv_out.size())) # 1*128*7*7 = 6272

    def forward(self, x):
        """ assumes x is 4D of shape BCHW, output will be 2D: B*n_actions """
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1) # flatten ? what this is not 1D ??!!
        return self.fc(conv_out)


if __name__ == '__main__':
    m1 = DQN((4,100,100), 5)
    print(m1)

    m2 = DuelingDQN((4,100,100), 5)
    print(m2)

    m3 = NoisyDQN((4,100,100), 5)
    print(m3)
