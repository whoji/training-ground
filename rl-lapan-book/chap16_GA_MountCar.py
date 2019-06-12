# https://github.com/openai/gym/wiki/Leaderboard
# https://github.com/openai/gym/wiki/MountainCar-v0

import gym
import numpy as np
import time
import copy

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

#MAX_BATCH_EPISODES = 100    # for training
#MAX_BATCH_STEPS = 10000     # for training

POPULATION_SIZE = 50        # number of policy net
ELITE_SIZE = 10             # number of top parents (elites) used to produce nextgen
NOISE_STD = 0.01            # std used to perturbate the weights (mutate)
LR = 0.001
HID_SIZE = 32
TRAIN_DONE_CUTOFF = 110
ENV_NAME = "MountainCar-v0"


class PolicyNet(nn.Module):
    def __init__(self, obs_size, action_size):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, action_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

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
        r = r + abs(s_new[1])
        ret += r
        steps += 1
        if terminal:
            break
        s = s_new
    return ret, steps

def get_mutated(net):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise_t = torch.from_numpy(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net

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
    writer = SummaryWriter(comment="-MtCar-GA")
    env = gym.make(ENV_NAME)

    gen_idx = 0
    nets = [PolicyNet(env.observation_space.shape[0], env.action_space.n
        ) for _ in range(POPULATION_SIZE)]
    population = [ (net, evaluate(env, net)) for net in nets ]


    while 1:
        population.sort(key=lambda p:p[1], reverse=True)
        Rets = [p[1][0] for p in population[:ELITE_SIZE]]
        Steps = [p[1][1] for p in population[:ELITE_SIZE]]
        Rets_mean, Rets_max, Rets_std = np.mean(Rets), np.max(Rets), np.std(Rets)
        Steps_mean, Steps_max, Steps_std = np.mean(Steps), np.max(Steps), np.std(Steps)

        writer.add_scalar("reward_mean", Rets_mean, gen_idx)
        writer.add_scalar("reward_std", Rets_std, gen_idx)
        writer.add_scalar("reward_max", Rets_max, gen_idx)
        print("gen-%d: reward_mean=%.2f, reward_max=%.2f, reward_std=%.2f | (%.2f, %.2f, %.2f, )"
            % (gen_idx, Rets_mean, Rets_max, Rets_std, Steps_mean, Steps_max, Steps_std))
        # print(Rets)
        if Steps_mean < TRAIN_DONE_CUTOFF:
            print("GGWP! done in %d gens" % gen_idx)
            break

        prev_population = population
        population = [population[0]]
        for _ in range(POPULATION_SIZE-1):
            parent_idx = np.random.randint(0, ELITE_SIZE)
            parent = prev_population[parent_idx][0]
            net = get_mutated(parent)
            fitness = evaluate(env,net)
            population.append((net, fitness))
        gen_idx += 1

    # render some runs of episodes
    watch_with_render(env, net=population[1][0], episodes=20, horizon=1000)
















