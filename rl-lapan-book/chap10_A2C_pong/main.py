import torch
import torch.nn as nn
import numpy as np
from model import AtariA2C

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50
REWARD_STEPS = 4
CLIP_GRAD = 0.1

def unpack_batch(batch, net, device = 'cpu'):
    s = []
    a = []
    r = []
    not_done_idx = []
    last_s = []

    for i, exp in enumerate(batch):
        s.append(np.array(exp.s, copy= False))
        a.append(exp.a)
        r.append(exp.r)
        if exp.last_state is not None:
            not_done_idx.append(i)
            last_s.append(np.array(exp.last_state), copy=False)

    s_v = torch.FloatTensor(s).to(device)
    a_t = torch.LongTensor(a).to(device)
    r_np = np.array(r, dtype = np.float32)

    if not_done_idx:
        last_s_v = torch.FloatTensor(last_s).to(device)
        last_V_v = net(last_s)[1]
        last_V_np = last_V_v.data.cpu().numpy()[:,0]
        r_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_V_np

    ref_V_v = torch.FloatTensor(r_np).to(device)
    return s_v, a_t, ref_V_v

if __name__ == '__main__':
    raise NotImplementedError