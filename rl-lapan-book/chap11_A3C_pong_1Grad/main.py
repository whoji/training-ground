import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import AtariA2C
import argparse
from tensorboardX import SummaryWriter
import ptan
import gym

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50
REWARD_STEPS = 4
CLIP_GRAD = 0.1
REWARD_CUTOFF = 19.5

def unpack_batch(batch, net, device = 'cpu'):
    s = []
    a = []
    r = []
    not_done_idx = []
    last_s = []

    for i, exp in enumerate(batch):
        s.append(np.array(exp.state, copy= False))
        a.append(exp.action)
        r.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(i)
            last_s.append(np.array(exp.last_state, copy=False))

    s_v = torch.FloatTensor(s).to(device)
    a_t = torch.LongTensor(a).to(device)
    r_np = np.array(r, dtype = np.float32)

    if not_done_idx:
        last_s_v = torch.FloatTensor(last_s).to(device)
        last_V_v = net(last_s_v)[1]
        last_V_np = last_V_v.data.cpu().numpy()[:,0]
        r_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_V_np

    ref_V_v = torch.FloatTensor(r_np).to(device)
    return s_v, a_t, ref_V_v

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c_"+args.name)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent,
        gamma=GAMMA, steps_count=REWARD_STEPS)

    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []
    uber_rewards = []
    uber_100 = -1000

    for i, exp in enumerate(exp_source):
        batch.append(exp)

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            uber_rewards.append(new_rewards[0])
            uber_100 = np.mean(uber_rewards[-100:])
            if  uber_100 > REWARD_CUTOFF:
                print("GGWP!!!")
                break

        if len(batch) < BATCH_SIZE:
            continue

        print("Training at %d-th step!! (last 100 reward: %.2f)" % (i, uber_100))

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

