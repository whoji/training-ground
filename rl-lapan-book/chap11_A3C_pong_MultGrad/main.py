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

REWARD_STEPS = 4
CLIP_GRAD = 0.1

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_CUTOFF = 18
PROC_COUNT = 4      # set to equal number of cpu cores
NUM_ENVS = 10       # each proc do this number of env. so 10 x 4 = 40

GRAD_BATCH = 64     # for each child proc to get the grad
TRAIN_BATCH = 2     # how many grad batches from the children procs to comb the grad


# TotalReward = collections.namedtuple('TotalReward', field_names='reward')

# make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

# to be executed in the children proc
def grads_func(proc_name, net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs,
        agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    batch = []
    i = 0 # frame index
    writer = SummaryWriter(comment = proc_name)
    uber_rewards = [-22]

    for exp in exp_source:
        i += 1
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            uber_rewards.append(new_rewards[0])
            if  np.mean(uber_rewards[-100:]) > REWARD_CUTOFF:
                print("GGWP!!! finished in %d steps" % i)
                break

        batch.append(exp)
        if len(batch) < GRAD_BATCH:
            continue

        print("[%s] Training at %d-th step!! (last 100 reward: %.2f)" % (
            proc_name, i, np.mean(uber_rewards[-100:])))

        s_v, a_t, vals_ref_v = unpack_batch(batch, net, device=device)
        batch.clear()

        net.zero_grad()
        policy_out_v, value_out_v = net(s_v)

        loss_value_v = F.mse_loss(value_out_v.squeeze(-1), vals_ref_v)

        log_prob_v = F.log_softmax(policy_out_v, dim=1)
        adv_v = vals_ref_v - value_out_v.detach() # A(s,a) = Q(s,a) - V(s)
        log_prob_actions_v = adv_v * log_prob_v[range(GRAD_BATCH), a_t]
        loss_policy_v = - log_prob_actions_v.mean()

        prob_v = F.softmax(policy_out_v, dim=1)
        entropy_v = - (prob_v * log_prob_v).sum(dim=1).mean()
        loss_entropy_v =  - ENTROPY_BETA * entropy_v

        loss_v = loss_value_v + loss_entropy_v + loss_policy_v # just for stats tracking purpose
        loss_v.backward()

        # writer / stats recording
        writer.add_scalar("entropy", entropy_v.item(), i)
        writer.add_scalar("loss_entropy", loss_entropy_v.item(), i)
        writer.add_scalar("loss_policy", loss_policy_v.item(), i)
        writer.add_scalar("loss_value", loss_value_v.item(), i)
        writer.add_scalar("loss_uber", loss_v.item(), i)
        # writer.add_scalar("grad_L2", np.sqrt(np.mean(np.square(grads))), i)
        # writer.add_scalar("grad_max", np.max(np.abs(grads)), i)
        # writer.add_scalar("grad_var", np.var(grads), i)

        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        grads = [p.grad.data.cpu().numpy() if p.grad is not None else None for p in net.parameters() ]
        train_queue.put(grads)

    writer.close()
    train_queue.put(None) # meaning this child proc has reached REWARD_CUTOFF, and training should stop


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    # device = torch.device("cuda" if args.cuda else "cpu")
    device = "cuda" if args.cuda else "cpu"
    # writer = SummaryWriter(comment="-pong-a3c_"+ NAME + "_" + args.name)

    temp_env = make_env()
    net = AtariA2C(temp_env.observation_space.shape, temp_env.action_space.n).to(device)
    net.share_memory() # cuda tensors are shared by default. but for cpu, need to do this
    print(net)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROC_COUNT)
    data_proc_list = []
    for proc_idx in range(PROC_COUNT):
        proc_name = "-a3c-grad_" + NAME + "_" + args.name + "#%d"%proc_idx
        data_proc = mp.Process(target=grads_func, args = (proc_name, net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break
            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for target_grad, grad in zip(grad_buffer, train_entry):
                    target_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    grad_v = torch.FloatTensor(grad).to(device)
                    param.grad = grad_v

                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                opt.step()
                grad_buffer = None





    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

