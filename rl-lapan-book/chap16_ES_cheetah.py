# modified from chap16_ES_CartPole

import gym
import roboschool
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
from tensorboardX import SummaryWriter
import collections

NOISE_STD = 0.01            # std used to perturbate the weights
LR = 0.01
HID_SIZE = 32
TRAIN_DONE_CUTOFF = 199

ITERS_PER_UPDATE = 10
MAX_ITERS = 1000
PROCESSES_COUNT = 4

RewardsItem = collections.namedtuple('RewardsItem', field_names=[
    'seed', 'pos_reward', 'neg_reward', 'steps'])

class Net(nn.Module):
    def __init__(self, obs_size, act_size, hid_size=64):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mu(x)

# taken from the ptan. so that i do not need to import ptan
def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

def evaluate(env, net, device="cpu"):
    '''
    evaluate performance on 1 whole episode
    a fitness function
    '''
    s = env.reset()
    ret = 0.0
    steps = 0
    while 1:
        # s_v = torch.FloatTensor([s]).to(device)
        s_v = default_states_preprocessor([s]).to(device)
        a_prob = net(s_v)
        #a = a_prob.max(dim=1)[1]
        s_new, r, terminal, _ = env.step(a_prob.data.cpu().numpy()[0])
        ret += r
        steps += 1
        if terminal:
            break
        s = s_new
    return ret, steps

def sample_noise(net,  device="cpu"):
    '''
    zero mean / unit variance, with size == net.parameters()
    mirrored sampling --> improve stability of convergence
    '''
    pos, neg = [], []
    for p in net.parameters():
        # noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32)).to(device)
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg

def eval_with_noise(env, net, noise, noise_std=NOISE_STD, device="cpu"):
    for p, p_n in zip(net.parameters(), noise):
        p.data += noise_std * p_n
    r, s = evaluate(env, net, device)
    for p, p_n in zip(net.parameters(), noise):
        p.data -= noise_std * p_n
    return r, s

def compute_centered_ranks(x):
    def compute_ranks(x):
        """
        Returns ranks in [0, len(x))
        Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
        """
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def train_step(opt, net, batch_noise, batch_reward, writer, step_idx, noise_std):
    weighted_noise = None
    norm_reward = compute_centered_ranks(np.array(batch_reward))

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n

    m_updates = []
    opt.zero_grad()
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * noise_std)
        p.grad = -update
        # m_updates.append(torch.norm(update).to(device))
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)
    opt.step()

def worker_func(worker_id, params_queue, reward_queue, device, noise_std):
    env = gym.make("RoboschoolHalfCheetah-v1")
    net = Net(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # print(net)
    net.eval()

    while 1:
        params = params_queue.get()
        if params is None:
            break
        net.load_state_dict(params)
        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = sample_noise(net, device=device)
            pos_reward, pos_steps = eval_with_noise(env, net, noise, noise_std, device=device)
            neg_reward, neg_steps = eval_with_noise(env, net, neg_noise, noise_std, device=device)
            reward_queue.put(RewardsItem(seed=seed,pos_reward=pos_reward,neg_reward=neg_reward,
                steps=pos_steps+neg_steps))

    pass

def watch_with_render(env, net, episodes, horizon):
    # import pdb; pdb.set_trace()
    for ep in range(episodes):
        s = env.reset()
        frames = 0
        for _ in range(horizon):
            env.render()
            #a = env.action_space.sample()
            s_v = torch.FloatTensor([s])
            # a_prob_v = nn.Softmax(dim=1)(net(s_v))
            # a_prob = a_prob_v.data.numpy()[0]
            # a = np.random.choice(len(a_prob), p = a_prob)
            a_prob = net(s_v)
            a = a_prob.max(dim=1)[1]
            a = a.data.numpy()[0] 

            a_prob = net(s_v)
            #a = a_prob.max(dim=1)[1]
            a = a_prob.data.cpu().numpy()[0]
            print(a)
            #a = a_prob.data.max(dim=1)[1].cpu().numpy()[0]

            s_new, r, terminal, _ = env.step(a)
            if terminal:
                print("finished %d/%d episode !! Frames=%d" % (ep, episodes, frames))
                frames = 0
                break
            else:
                frames += 1
                s = s_new
    env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA mode")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    writer = SummaryWriter(comment="-cheetah-es_lr=%.3e_sigma=%.3e" % (args.lr, args.noise_std))
    env = gym.make("RoboschoolHalfCheetah-v1")
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)
  
    # watch_with_render(env, net, episodes=20, horizon=1000)

    params_queues = [mp.Queue(maxsize=1) for _ in range(PROCESSES_COUNT)]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    workers = []

    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker_func, args=(idx, params_queue, rewards_queue,
            device, args.noise_std))
        proc.start()
        workers.append(proc)
    print("All procs / workers started !!!!")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)


    for step_idx in range(args.iters):
        # broadcast network params
        params = net.state_dict()
        for q in params_queues:
            q.put(params)

        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        batch_steps_data = [] # ?
        results_counter = 0 # counter for the end_of_run condition

        while 1:
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed)    
                noise, neg_noise = sample_noise(net)
                batch_noise.append(noise)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.pos_reward)
                batch_reward.append(reward.neg_reward)
                results_counter += 1
                batch_steps += reward.steps
                batch_steps_data.append(reward.steps) # ?

            if results_counter >= PROCESSES_COUNT * ITERS_PER_UPDATE:
                break
            
            time.sleep(0.01) # ? 


        dt_data = time.time() - t_start
        m_reward = np.mean(batch_reward)

        train_step(opt, net, batch_noise, batch_reward, writer, step_idx, args.noise_std)
        
        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        dt_step = time.time() - t_start - dt_data

        print("%d: reward=%.2f, speed=%.2f f/s, data_gather=%.3f, train=%.3f, steps_mean=%.2f, min=%.2f, max=%.2f, steps_std=%.2f" % (
            step_idx, m_reward, speed, dt_data, dt_step, np.mean(batch_steps_data),
            np.min(batch_steps_data), np.max(batch_steps_data), np.std(batch_steps_data)))

    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()


    # render some runs of episodes
    watch_with_render(env, net, episodes=100, horizon=1000)














