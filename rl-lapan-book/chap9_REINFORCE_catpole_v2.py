# code modified from chap4_CE_CartPole
# detailed maths. check this link
# http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from tensorboardX import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01 #0.0005
EPISODE_TO_TRAIN = 4
ENV_NAME = "CartPole-v0"
HIDDEN_SIZE = 128    # width of the NN
STOP_CRITERIA = 195

Episode = namedtuple('Episode', field_names=['return_','steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['s', 'a', 'r'])

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_action):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_action)
        )

    def forward(self, x):
        return self.net(x)


def process_episode(episode_steps):
    s = [e.s for e in episode_steps]
    a = [e.a for e in episode_steps]
    r = [e.r for e in episode_steps]
    n = len(episode_steps)
    for i in reversed(range(n-1)):
        # import pdb; pdb.set_trace()
        r[i] = r[i] + r[i+1]*GAMMA
    return (s, a, r)

def iterate_episodes(env, net):
    episode_steps = []
    s = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([s])
        nn_output_v = net(obs_v)
        act_probs_v = sm(nn_output_v)
        act_probs = act_probs_v.data.numpy()[0]
        a = np.random.choice(len(act_probs), p=act_probs)
        s_new, r, terminal, _ = env.step(a)
        episode_steps.append(EpisodeStep(s=s,a=a,r=r))

        if terminal:
            S, A, R = process_episode(episode_steps)
            episode = (S, A, R)
            yield episode
            episode_steps = []
            s_new = env.reset()

        s = s_new


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    net = Net(s_size, HIDDEN_SIZE, a_size)
    print(net)
    opt = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter()

    batch_size = 0
    batch_s, batch_a, batch_r = [], [], []

    stats_returns = []

    for i , (s, a, r) in enumerate(iterate_episodes(env, net)):
        # import pdb; pdb.set_trace()
        batch_s += s
        batch_a += a
        batch_r += r
        batch_size += 1

        Ret = len(s)
        stats_returns.append(Ret)

        if i % 10 ==0:  print("%d th ep | len:%d | mean:%.2f | max:%.2f" %
            (i, Ret, np.mean(stats_returns[-100:]),np.max(stats_returns[-100:]) ))

        if np.mean(stats_returns[-100:]) >= STOP_CRITERIA:
            print("GG solved in %d episodes !!" % i)
            break

        if batch_size < EPISODE_TO_TRAIN:
            continue

        # train the network
        opt.zero_grad()
        s_v = torch.FloatTensor(batch_s)
        a_t = torch.LongTensor(batch_a)
        r_v = torch.FloatTensor(batch_r)
        net_output = net(s_v)
        log_prob_v = F.log_softmax(net_output, dim=1)
        log_prob_actions_v = r_v * log_prob_v[range(len(a_t)), a_t]
        loss_v = - log_prob_actions_v.mean()
        loss_v.backward()
        opt.step()

        # clear the batch
        batch_size = 0
        batch_s, batch_a, batch_r = [], [], []

    writer.close()