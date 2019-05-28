import gym
import torch
import numpy as np
import argparse
import time

import wrapper
from model import DQN
from agent import ExperienceBuffer, Agent, calc_loss

ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="model file")
    parser.add_argument("-e", "--env", default=ENV_NAME, help="env name")
    parser.add_argument("-r", "--record", default=False, help="dir to save video")
    parser.add_argument("-t", "--gpu2cpu", default=True, help="if model is cuda, and load to cpu-only")
    parser.add_argument("-rd", "--random_action", default=False, help="let it do random action")
    args = parser.parse_args()

    env = wrapper.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = DQN(env.observation_space.shape, env.action_space.n)
    if args.gpu2cpu:
        #net.load_state_dict(torch.load(args.model), map_location=lambda storage, loc: storage)
        # net.load_state_dict(torch.load(args.model, map_location={'cuda:0':'cpu'}))
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("model_loaded (cuda->cpu)")
    else:
        net.load_state_dict(torch.load(args.model))
        print("model_loaded")

    s = env.reset()
    total_reward = 0.0
    while 1:
        start_ts = time.time()
        env.render()
        s_v = torch.tensor(np.array([s], copy=False))
        q_vals = net(s_v).data.numpy()[0]
        if args.random_action:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_vals)

        s, r, terminal, _ = env.step(action)
        total_reward += r
        if terminal:
            break

        delta = 1/ FPS - (time.time()-start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
