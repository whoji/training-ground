import torch
import torch.nn as nn
import numpy as np

GAMMA = 0.99
REWARD_STEPS = 4

class AtariA2C(nn.Module):
    """docstring for AtariA2C"""
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4), # in / out / kernel / stride
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, 1),
            nn.ReLU()
        )

        conv_out_size = self.__get_conv_out(input_shape)

        # actor head
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # critic head
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def __get_conv_out(self, input_shape):
        dummy_conv_out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(dummy_conv_out.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device = 'cpu'):
    s = []
    a = []
    r = []
    not_done_idx = []
    last_s = []

    for i, exp in enumerate(batch):
        s.append(np.array(exp.state, copy= False))
        a.append(exp.action)
        r.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(i)
            last_s.append(np.array(exp.last_state, copy=False))

    s_v = torch.FloatTensor(s).to(device)
    a_t = torch.LongTensor(a).to(device)
    r_np = np.array(r, dtype = np.float32)

    if not_done_idx:
        last_s_v = torch.FloatTensor(last_s).to(device)
        last_V_v = net(last_s_v)[1]
        last_V_np = last_V_v.data.cpu().numpy()[:,0]
        r_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_V_np

    ref_V_v = torch.FloatTensor(r_np).to(device)
    return s_v, a_t, ref_V_v
