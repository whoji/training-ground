import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import numpy as np
from agent import float32_preprocessor, AgentA2C
from model import A2CModel
import gym
import pybullet_envs
from tensorboardX import SummaryWriter

PIE = 3.1415926
ENV_NAME = 'MinitaurBulletEnv-v0'
GAMMA = 0.99
LEARNING_RATE = 5e-5 # 0.001
ENTROPY_BETA = 1e-4 # 0.01
BATCH_SIZE = 32 # 128
REWARD_STEPS = 2 #4
TEST_ITERS = 1000
#CLIP_GRAD = 0.1
STOP_CRITERIA = 2000

def test_net(net, env, episodes=10, device='cpu'):
    all_Ret = []
    all_steps = []
    for i in range(episodes):
        Ret = 0.0
        steps = 0
        s = env.reset()
        while 1:
            s_v = float32_preprocessor([s]).to(device)
            mu_v, _ , _ = net(s_v)
            # now take the mean as the action.
            a = mu_v.squeeze(dim=0).data.cpu().numpy()
            a = np.clip(a, -1, 1)
            s_new, r, terminal, _ = env.step(a)
            Ret += r
            steps += 1
            if terminal:
                print("ep %d | steps: %d | return: %.2f" % (i, steps, Ret))
                Ret, steps = 0.0, 0
                all_Ret.append(Ret)
                all_steps.append(steps)
                break
    return np.mean(all_Ret), np.mean(all_steps)

def calc_logprob(mu_v, var_v, actions_v):
    # book_page404: log(policy(a|s)) = log (gaussian formula here)
    part1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    part2 = - torch.log(torch.sqrt(2 * PIE * var_v)) # entropy term.
    return part1 + part2


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.shape[0]

    net = A2CModel(s_size, a_size)
    print(net)

    agent = AgentA2C(net)

    # watch the performance before any training
    # agent.watch_with_render_continous(ENV_NAME, 5, 10000)

    opt =  opt.Adam(params=net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment="-MinitaurBulletEnv-A2C-0625")

    batch_size = 0
    batch_s, batch_a, batch_r, batch_v = [], [], [], []
    reward_sum = 0.0
    episodes_returns = []

    for i , (s, a, r, Ret) in enumerate(agent.iterate_sample(env, REWARD_STEPS)):
        # import pdb; pdb.set_trace()
        reward_sum += r
        batch_s.append(s)
        batch_a.append(a)
        batch_r.append(r)

        if Ret: episodes_returns.append(Ret)

        if i % 1000 ==0 and len(episodes_returns) > 0:
            print("%d th step (sample) | ret.mean:%.2f | ret.max:%.2f" %
                (i, np.mean(episodes_returns[-100:]),np.max(episodes_returns[-100:]) ))

        if np.mean(episodes_returns[-100:]) >= STOP_CRITERIA:
            print("GG solved in %s step | %d episodes !!" % (i, len(episodes_returns)))
            break

        if len(batch_s) < BATCH_SIZE:
            continue

        s_v = torch.FloatTensor(batch_s)
        a_t = torch.FloatTensor(batch_a)
        r_v = torch.FloatTensor(batch_r)

        # train the network | ACTOR
        opt.zero_grad()
        mu_v, var_v, value_v = net(s_v)

        loss_value_v = F.mse_loss(value_v.squeeze(-1), r_v)

        # A(s,a) = Q(s,a) - V(s)
        adv_v = r_v.unsqueeze(dim=-1) - value_v.detach()

        log_prob_v = adv_v * calc_logprob(mu_v, var_v, a_t)
        loss_policy_v = - log_prob_v.mean()

        entropy_v = -  ((torch.log(2*PIE*var_v)+1)/2)  .mean()
        loss_entropy_v =  ENTROPY_BETA * entropy_v

        loss_v = loss_policy_v + loss_entropy_v + loss_value_v
        loss_v.backward()
        opt.step()

        # clear the batch
        batch_s, batch_a, batch_r = [], [], []

        # writer / stats recording
        writer.add_scalar("loss_entropy", loss_entropy_v.item(), i)
        writer.add_scalar("loss_policy", loss_policy_v.item(), i)
        writer.add_scalar("loss_value", loss_value_v.item(), i)
        writer.add_scalar("loss_uber", loss_v.item(), i)

    writer.close()

    # render some runs of episodes
    #watch_with_render(env, net, episodes=20, horizon=1000)
