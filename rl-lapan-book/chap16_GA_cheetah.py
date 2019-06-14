# modified from chap16_ES_cheetah

import gym
import roboschool
import copy
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
from tensorboardX import SummaryWriter
import collections

POPULATION_SIZE = 2000      # number of policy net
ELITE_SIZE = 10             # number of top parents (elites) used to produce nextgen
NOISE_STD = 0.01            # std used to perturbate the weights (mutate)
LR = 0.01
WORKER_COUNT = 6
SEEDS_PER_WORKER = POPULATION_SIZE // WORKER_COUNT
MAX_SEED = 2**32 - 1


OutputItem = collections.namedtuple('OutputItem', field_names=[
    'seeds', 'reward', 'steps'])

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


def get_mutated(net, seed, copy_net=True):
    new_net = copy.deepcopy(net) if copy_net else net
    np.random.seed(seed)
    for p in new_net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net

def build_net(env, seeds):
    torch.manual_seed(seeds[0])
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    for seed in seeds[1:]:
        net = get_mutated(net, seed, copy_net=False)
    return net

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

def worker_func(input_queue, output_queue):
    env = gym.make("RoboschoolHalfCheetah-v1")
    cache = {}

    while 1:
        parents = input_queue.get()
        if parents is None:
            break
        new_cache = {}
        for net_seeds in parents:
            if len(net_seeds) > 1:
                net = cache.get(net_seeds[:-1])
                if net is not None:
                    net = get_mutated(net, net_seeds[-1])
                else: 
                    net = build_net(env, net_seeds)
            else:
                net = build_net(env, net_seeds)
            new_cache[net_seeds] = net
            reward, steps = evaluate(env, net)
            output_queue.put(OutputItem(seeds=net_seeds, reward=reward, steps=steps))
        cache = new_cache

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
    args = parser.parse_args()

    writer = SummaryWriter(comment="-cheetah-es_lr=%.3e_sigma=%.3e" % (args.lr, args.noise_std))

    # try:
    #     net.load_state_dict(torch.load('./cheetah_es_model.pth'))
    #     print("OLD MODEL LOADED")
    # except:
    #     print("no old model found, using new init model.")
  
    # watch_with_render(env, net, episodes=2, horizon=1000)

    input_queues = []
    output_queue = mp.Queue(maxsize=WORKER_COUNT)
    workers =[]

    for _ in range(WORKER_COUNT):
        input_queue = mp.Queue(maxsize=1)
        input_queues.append(input_queue)
        worker = mp.Process(target=worker_func, args=(input_queue, output_queue))
        worker.start()
        seeds =  [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        input_queue.put(seeds)

    gen_idx = 0
    elite = None

    while True:
        t_start = time.time()
        batch_steps = 0
        population = []
        while len(population) < SEEDS_PER_WORKER * WORKER_COUNT:
            out_item = output_queue.get()
            population.append((out_item.seeds, out_item.reward))
            batch_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        population.sort(key=lambda p: p[1], reverse=True)
        rewards = [p[1] for p in population[:ELITE_SIZE]]
        reward_mean = np.mean(rewards)
        reward_max = np.max(rewards)
        reward_std = np.std(rewards)
        writer.add_scalar("reward_mean", reward_mean, gen_idx)
        writer.add_scalar("reward_std", reward_std, gen_idx)
        writer.add_scalar("reward_max", reward_max, gen_idx)
        writer.add_scalar("batch_steps", batch_steps, gen_idx)
        writer.add_scalar("gen_seconds", time.time() - t_start, gen_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, gen_idx)
        print("%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f, speed=%.2f f/s" % (
            gen_idx, reward_mean, reward_max, reward_std, speed))

        elite = population[0]
        for worker_queue in input_queues:
            seeds = []
            for _ in range(SEEDS_PER_WORKER):
                parent = np.random.randint(ELITE_SIZE)
                next_seed = np.random.randint(MAX_SEED)
                seeds.append(tuple(list(population[parent][0]) + [next_seed]))
            worker_queue.put(seeds)
        gen_idx += 1


    # save the model for later use
    # # torch.save(net.state_dict(),"./cheetah_model.pth")
    # net.save_state_dict('./cheetah_es_model.pth')

    # # render some runs of episodes
    # watch_with_render(env, net, episodes=2, horizon=1000)

    # import pdb; pdb.set_trace()














