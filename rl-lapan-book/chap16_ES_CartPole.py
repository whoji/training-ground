import gym
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

MAX_BATCH_EPISODES = 100    # for training
MAX_BATCH_STEPS = 10000     # for training
NOISE_STD = 0.01            # std used to perturbate the weights
LR = 0.001
HID_SIZE = 32
TRAIN_DONE_CUTOFF = 199


class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

## old
def evaluate(env, net):
    '''
    evaluate performance on 1 whole episode
    a fitness function
    '''
    s = env.reset()
    ret = 0.0
    steps = 0
    while 1:
        s_v = torch.FloatTensor([s])
        a_prob = net(s_v)
        a = a_prob.max(dim=1)[1]
        s_new, r, terminal, _ = env.step(a.data.numpy()[0])
        ret += r
        steps += 1
        if terminal:
            break
        s = s_new
    return ret, steps

def sample_noise(net):
    '''
    zero mean / unit variance, with size == net.parameters()
    mirrored sampling --> improve stability of convergence
    '''
    pos, neg = [], []
    for p in net.parameters():
        # noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg

def eval_with_noise(env, net, noise):
    old_params = net.state_dict()
    for p, p_n in zip(net.parameters(), noise):
        p.data += NOISE_STD * p_n
    ret, steps = evaluate(env, net)
    # restore the old net
    net.load_state_dict(old_params)
    return ret, steps

def train_step(net, batch_noise, batch_reward, writer, step_idx):
    # normalize the rewards (ret)
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s

    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n

    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LR * update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)


if __name__ == '__main__':
    writer = SummaryWriter(comment="-cartpole-es")
    env = gym.make("CartPole-v0")
    net = Net(env.observation_space.shape[0], env.action_space.n)
    print(net)

    step_idx = 0
    while 1:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            noise, neg_noise = sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)

            reward, steps = eval_with_noise(env, net, noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = eval_with_noise(env, net, neg_noise)
            batch_reward.append(reward)
            batch_steps += steps

            if batch_steps > MAX_BATCH_STEPS:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > TRAIN_DONE_CUTOFF:
            print("GGWP! done in %d steps" % step_idx)
            break

        train_step(net, batch_noise, batch_reward, writer, step_idx)

        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, m_reward, speed))















