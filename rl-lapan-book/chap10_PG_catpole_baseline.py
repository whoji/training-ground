# code modified from chap9_PG_catpole.py. now adding the --baseline portion
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

ENV_NAME = "CartPole-v0"
GAMMA = 0.99
LEARNING_RATE = 0.001 #0.0005
# EPISODE_TO_TRAIN = 4
HIDDEN_SIZE = 128    # width of the NN
STOP_CRITERIA = 195
ENTROPY_BETA = 0.01
BATH_SIZE = 8
REWARD_STEPS = 10 # specifiy how many steps ahead the Bellman equiation is
                  # unrolled to estimate the discounted total reward of every transition
BASELINE = True

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


def process_episode(episode_steps, steps = REWARD_STEPS):
    s = [e.s for e in episode_steps]
    a = [e.a for e in episode_steps]
    r = [e.r for e in episode_steps]
    n = len(episode_steps)
    q = []
    for i in range(n):
        # import pdb; pdb.set_trace()
        r_roll = 0.0
        for j in range(steps):
            if i+j >= n:
                break
            r_roll += r[i+j] * GAMMA**j
        q.append(r_roll)
    assert len(q) == len(r)
    # import pdb; pdb.set_trace()
    return (s, a, q)

def iterate_sample(env, net, steps = REWARD_STEPS):
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
            for i in range(len(S)):
                Ret = len(S) if i == len(S)-1  else None
                yield (S[i], A[i], R[i], Ret)
            episode_steps = []
            s_new = env.reset()

        s = s_new

def watch_with_render(env, net, episodes, horizon):
    # import pdb; pdb.set_trace()
    for ep in range(episodes):
        s = env.reset()
        frames = 0
        for _ in range(horizon):
            env.render()
            #a = env.action_space.sample()
            s_v = torch.FloatTensor([s])
            a_prob_v = nn.Softmax(dim=1)(net(s_v))
            a_prob = a_prob_v.data.numpy()[0]
            a = np.random.choice(len(a_prob), p = a_prob)
            s_new, r, terminal, _ = env.step(a)
            if terminal:
                print("finished %d/%d episode !! Frames=%d" % (ep, 20, frames))
                frames = 0
                break
            else:
                frames += 1
                s = s_new
    env.close()


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    net = Net(s_size, HIDDEN_SIZE, a_size)
    print(net)
    opt = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment="-cartpole-pg-0530" + "-baseline=%s" % BASELINE)

    batch_size = 0
    batch_s, batch_a, batch_r = [], [], []
    reward_sum = 0.0
    episodes_returns = []

    for i , (s, a, r, Ret) in enumerate(iterate_sample(env, net, REWARD_STEPS)):
        # import pdb; pdb.set_trace()
        reward_sum += r
        baseline = reward_sum / (i + 1)
        batch_s.append(s)
        batch_a.append(a)
        if BASELINE:
            batch_r.append(r-baseline)
        else:
            batch_r.append(r)

        if Ret: episodes_returns.append(Ret)

        if i % 1000 ==0 and len(episodes_returns) > 0:
            print("%d th step (sample) | ret.mean:%.2f | ret.max:%.2f" %
                (i, np.mean(episodes_returns[-100:]),np.max(episodes_returns[-100:]) ))

        if np.mean(episodes_returns[-100:]) >= STOP_CRITERIA:
            print("GG solved in %s step | %d episodes !!" % (i, len(episodes_returns)))
            break

        if len(batch_s) < BATH_SIZE:
            continue

        # train the network
        opt.zero_grad()
        s_v = torch.FloatTensor(batch_s)
        a_t = torch.LongTensor(batch_a)
        r_v = torch.FloatTensor(batch_r)
        net_output = net(s_v)
        log_prob_v = F.log_softmax(net_output, dim=1)
        log_prob_actions_v = r_v * log_prob_v[range(len(a_t)), a_t]
        loss_policy_v = - log_prob_actions_v.mean()

        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate(
            [p.grad.data.numpy().flatten() for p in net.parameters() if p.grad is not None])

        prob_v = F.softmax(net_output, dim=1)
        entropy_v = - (prob_v * log_prob_v).sum(dim=1).mean()
        loss_entropy_v = - ENTROPY_BETA * entropy_v
        loss_entropy_v.backward()

        loss_v = loss_policy_v + loss_entropy_v

        opt.step()

        # clear the batch
        batch_s, batch_a, batch_r = [], [], []

        # writer / stats recording

        writer.add_scalar("baseline", baseline, i)
        writer.add_scalar("entropy", entropy_v.item(), i)
        writer.add_scalar("batch_scales", np.mean(batch_r), i)
        writer.add_scalar("loss_entropy", loss_entropy_v.item(), i)
        writer.add_scalar("loss_policy", loss_policy_v.item(), i)
        writer.add_scalar("loss_total", loss_v.item(), i)

        writer.add_scalar("grad_L2", np.sqrt(np.mean(np.square(grads))), i)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), i)
        writer.add_scalar("grad_var", np.var(grads), i)

    writer.close()

    # render some runs of episodes
    watch_with_render(env, net, episodes=20, horizon=1000)
