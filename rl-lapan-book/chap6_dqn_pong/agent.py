import numpy as np
import collections
import torch
import torch.nn as nn

Exp = collections.namedtuple('Exp', field_names=['s','a','r','terminal','s_new'])

class ExperienceBuffer:
    def __init__(self, capactiy):
        self.buffer = collections.deque(maxlen=capactiy)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, terminal, s_new = zip(*[self.buffer[i] for i in idx])
        return np.array(s), np.array(a), np.array(r, dtype=np.float32
            ), np.array(terminal, dtype=np.uint8), np.array(s_new)


class Agent(object):
    """docstring for Agent"""
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        self.state = None
        self.total_reward = 0.0

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        terminal_reward = None

        if np.random.random() < epsilon:
            a = self.env.action_space.sample()
        else:
            s_a = np.array([self.state], copy=False)
            s_v = torch.tensor(s_a).to(device)
            q_v = net(s_v)
            _, a_v = torch.max(q_v, dim=1)
            a = int(a_v.item())

        s_new, r, terminal, _ = self.env.step(a)
        self.total_reward += r
        exp = Exp(self.state, a, r, terminal, s_new)
        self.exp_buffer.append(exp)
        self.state = s_new
        if terminal:
            terminal_reward = self.total_reward
            self._reset()
        return terminal_reward


def calc_loss(batch, net, tgt_net, device="cpu", gamma = 0.99):
    s, a, r, terminal, s_new = batch

    s_v = torch.tensor(s).to(device)
    s_new_v = torch.tensor(s_new).to(device)
    a_v = torch.LongTensor(a).to(device)
    r_v = torch.tensor(r).to(device)
    terminal_mask = torch.ByteTensor(terminal).to(device)

    state_action_val = net(s_v).gather(1, a_v.unsqueeze(-1)).squeeze(-1)
    next_state_val = tgt_net(s_new_v).max(1)[0]
    next_state_val[terminal_mask] = 0.0
    next_state_val = next_state_val.detach()
    expected_state_action_val = next_state_val * gamma + r_v

    loss = nn.MSELoss()(state_action_val, expected_state_action_val)
    return loss
