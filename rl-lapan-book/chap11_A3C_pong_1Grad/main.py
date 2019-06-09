# note: cannot run the cuda version of this on a Windows OS
# https://pytorch.org/docs/stable/notes/windows.html#cuda-ipc-operations

import numpy as np
import argparse
import ptan
import gym
import collections
from model import AtariA2C, unpack_batch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_CUTOFF = 18
PROC_COUNT = 4      # set to equal number of cpu cores
NUM_ENVS = 10       # each proc do this number of env. so 10 x 4 = 40

TotalReward = collections.namedtuple('TotalReward', field_names='reward')

# make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

# to be executed in the children proc
def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs,
        agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    # device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    writer = SummaryWriter(comment="-pong-a3c_"+ NAME + "_" + args.name)

    temp_env = make_env()
    net = AtariA2C(temp_env.observation_space.shape, temp_env.action_space.n).to(device)
    net.share_memory() # cuda tensors are shared by default. but for cpu, need to do this
    print(net)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROC_COUNT)
    data_proc_list = []
    for _ in range(PROC_COUNT):
        data_proc = mp.Process(target=data_func, args = (net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    i = 0
    uber_rewards = [-22]

    while True:
        train_entry = train_queue.get()
        if isinstance(train_entry, TotalReward):
            uber_rewards.append(train_entry.reward)
            if  np.mean(uber_rewards[-100:]) > REWARD_CUTOFF:
                print("GGWP!!! finished in %d steps" % i)
                break
            continue

        i += 1
        batch.append(train_entry)
        if len(batch) < BATCH_SIZE:
            continue

        print("Training at %d-th step!! (last 100 reward: %.2f)" % (i, np.mean(uber_rewards[-100:])))

        s_v, a_t, vals_ref_v = unpack_batch(batch, net, device=device)
        batch.clear()

        opt.zero_grad()
        policy_out_v, value_out_v = net(s_v)

        loss_value_v = F.mse_loss(value_out_v.squeeze(-1), vals_ref_v)

        log_prob_v = F.log_softmax(policy_out_v, dim=1)
        adv_v = vals_ref_v - value_out_v.detach() # A(s,a) = Q(s,a) - V(s)
        log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), a_t]
        loss_policy_v = - log_prob_actions_v.mean()

        prob_v = F.softmax(policy_out_v, dim=1)
        entropy_v = - (prob_v * log_prob_v).sum(dim=1).mean()
        loss_entropy_v =  - ENTROPY_BETA * entropy_v

        # loss_policy_v = loss_policy_v + loss_entropy_v
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten() for p in net.parameters() if p.grad is not None])

        loss_value_v = loss_value_v + loss_entropy_v
        loss_value_v.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        opt.step()
        loss_v = loss_value_v + loss_entropy_v + loss_policy_v # just for stats tracking purpose

        # writer / stats recording
        writer.add_scalar("entropy", entropy_v.item(), i)
        writer.add_scalar("loss_entropy", loss_entropy_v.item(), i)
        writer.add_scalar("loss_policy", loss_policy_v.item(), i)
        writer.add_scalar("loss_value", loss_value_v.item(), i)
        writer.add_scalar("loss_uber", loss_v.item(), i)

        writer.add_scalar("grad_L2", np.sqrt(np.mean(np.square(grads))), i)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), i)
        writer.add_scalar("grad_var", np.var(grads), i)

    writer.close()

    for p in data_proc_list:
        p.terminate()
        p.join()

