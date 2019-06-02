import numpy as np
import collections
import torch
import torch.nn as nn

PRIORITY_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_DECAY_STEPS = 10000

class PriorityReplayBuffer(object):
    """docstring for PriorityReplayBuffer"""
    def __init__(self, capacity, prob_alpha=PRIORITY_REPLAY_ALPHA, beta=BETA_START):
        self.prob_alpha = prob_alpha
        self.beta = beta
        self.capacity = capacity
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        if len(self) == 0:
            max_priority = 1.0
        else:
            max_priority = self.priorities.max()

        if len(self) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

        self.beta = min(1.0, self.beta + (1.0-BETA_START)/BETA_DECAY_STEPS)

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probs = priorities ** self.prob_alpha # element-wise
        probs = probs / probs.sum()
        idx = np.random.choice(len(self), batch_size, p=probs)
        # samples = [self.buffer[i] for i in idx]
        s, a, r, terminal, s_new = zip(*[self.buffer[i] for i in idx])
        samples = ( np.array(s), np.array(a), np.array(r, dtype=np.float32
            ), np.array(terminal, dtype=np.uint8), np.array(s_new)
        )

        weights = (len(self) * probs[idx]) ** (- self.beta)
        weights = weights / weights.max()

        return samples, idx, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for i, priority in zip(batch_indices, batch_priorities):
            self.priorities[i] = priority


def calc_loss_weighted(batch, batch_w, net, tgt_net, gamma, device= "cpu"):
    #import pdb; pdb.set_trace()
    s, a, r, terminal, s_new = batch

    s_v     = torch.tensor(s).to(device)
    s_new_v = torch.tensor(s_new).to(device)
    a_v     = torch.LongTensor(a).to(device)
    r_v     = torch.tensor(r).to(device)
    terminal_mask = torch.ByteTensor(terminal).to(device)
    w_v     = torch.tensor(batch_w).to(device)

    q_s_a = net(s_v).gather(1, a_v.unsqueeze(-1)).squeeze(-1)
    v_s_new = tgt_net(s_new_v).max(1)[0]
    v_s_new[terminal_mask] = 0.0

    expected_q_s_a = v_s_new.detach() * gamma + r_v
    losses_v = w_v * (q_s_a - expected_q_s_a) ** 2
    return losses_v.mean(), losses_v + 1e-5


