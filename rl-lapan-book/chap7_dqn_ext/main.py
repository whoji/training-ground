import wrapper
from model import DQN, DuelingDQN
from model_noisy import NoisyDQN
from agent import ExperienceBuffer, Agent, calc_loss

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_CUTOFF = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000 # UNIT: number of sars'
REPLAY_START_SIZE = 10000
LEARNING_RATE = 0.0001
SYNC_tARGET_FRAME = 1000 # when q and q' will sync

EPSILON_DECAY_STEPS = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.05

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enables CUDA")
    parser.add_argument("--env",  default=ENV_NAME, help="Name of the gym env.")
    parser.add_argument("--reward", default=MEAN_REWARD_CUTOFF, type=float, help="When to stop training")
    parser.add_argument("--double", default=False, action="store_true", help="DDQN")
    parser.add_argument("--duel",   default=False, action="store_true", help="DuelingDQN")
    parser.add_argument("--noisy",   default=False, action="store_true", help="NoisyDQN")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    print("WE ARE RUNNING ON [ %s ]" % device)
    writer = SummaryWriter(comment='-' + args.env)

    env = wrapper.make_env(args.env)
    if args.duel:
        net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    elif args.noisy:
        net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    print(net)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    #import pdb; pdb.set_trace()
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    opt = optim.Adam(net.parameters(),lr=LEARNING_RATE)
    Rets = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_Ret = -9999

    while 1:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_STEPS)

        r = agent.play_step(net, epsilon, device=device)
        if r is not None:
            # when end of 1 episode
            Rets.append(r)
            speed = (frame_idx-ts_frame) / (time.time()-ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_Ret = np.mean(Rets[-100:])

            print("%d th step | %d episodes | mean Ret: %.3f | eps %.3f | speed %.3f f/s" %
                (frame_idx, len(Rets), mean_Ret, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("ret_100", mean_Ret, frame_idx)
            writer.add_scalar("ret_cur", r, frame_idx)

            if mean_Ret > best_mean_Ret:
                torch.save(net.state_dict(), args.env + "-best.dat")
                print("best_mean_Ret updated: %.3f --> %.3f" %
                    (best_mean_Ret, mean_Ret))
                best_mean_Ret = mean_Ret
                if best_mean_Ret > MEAN_REWARD_CUTOFF:
                    print("GGWP. solved in %d episode | %d frames " %
                        (len(Rets), frame_idx))
                    break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_tARGET_FRAME == 0:
            tgt_net.load_state_dict(net.state_dict())

        opt.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device, GAMMA, args.double)
        loss_t.backward()
        opt.step()
