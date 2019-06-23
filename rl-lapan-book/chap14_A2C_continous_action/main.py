import torch
import torch.nn as nn
import numpy as np
from agent import float32_preprocessor, AgentA2C
from model import A2CModel

PIE = 3.1415926
ENV_NAME = 'MinitaurBulletEnv-v0'
GAMMA = 0.99
LEARNING_RATE = 5e-5 # 0.001
ENTROPY_BETA = 1e-4 # 0.01
BATCH_SIZE = 32 # 128
REWARD_STEPS = 2 #4
TEST_ITERS = 1000
#CLIP_GRAD = 0.1
#REWARD_CUTOFF = 19.5

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
            s_new, r, terminal, _ = env.step(action)
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


def iterate_sample(env, steps, actor_net , critic_net):
    episode_steps = []
    s = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([s])
        nn_output_v = actor_net(obs_v)
        act_probs_v = sm(nn_output_v)
        act_probs = act_probs_v.data.numpy()[0]
        a = np.random.choice(len(act_probs), p=act_probs)
        s_new, r, terminal, _ = env.step(a)
        episode_steps.append(EpisodeStep(s=s,a=a,r=r))

        if terminal:
            S, A, R, V = process_episode(episode_steps)
            for i in range(len(S)):
                Ret = len(S) if i == len(S)-1  else None
                # pdb.set_trace()
                yield (S[i], A[i], R[i], V[i], Ret)
            episode_steps = []
            s_new = env.reset()

        s = s_new

def watch_with_render(env, net, episodes, horizon):
    # import pdb; pdb.set_trace()
    for ep in range(episodes):
        s = env.reset()
        frames = 0
        for _ in range(horizon):
            env.render()
            #a = env.action_space.sample()
            s_v = torch.FloatTensor([s])
            a_prob_v = nn.Softmax(dim=1)(net(s_v))
            a_prob = a_prob_v.data.numpy()[0]
            a = np.random.choice(len(a_prob), p = a_prob)
            s_new, r, terminal, _ = env.step(a)
            if terminal:
                print("finished %d/%d episode !! Frames=%d" % (ep, 20, frames))
                frames = 0
                break
            else:
                frames += 1
                s = s_new
    env.close()


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    net = A2CModel(s_size, HIDDEN_SIZE, a_size)
    print(net)

    opt =  optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(comment="-cartpole-PG-1net-0603")

    batch_size = 0
    batch_s, batch_a, batch_r, batch_v = [], [], [], []
    reward_sum = 0.0
    episodes_returns = []

    for i , (s, a, r, Ret) in enumerate(iterate_sample(env, REWARD_STEPS, net)):
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

        if len(batch_s) < BATH_SIZE:
            continue

        s_v = torch.FloatTensor(batch_s)
        a_t = torch.LongTensor(batch_a)
        r_v = torch.FloatTensor(batch_r)

        # train the network | ACTOR
        opt.zero_grad()
        mu_v, var_v, value_v = net(s_v)

        loss_value_v = F.mse_loss(value_v, r_v)

        # A(s,a) = Q(s,a) - V(s)
        adv_v = r_v.unsqeeze(dim=-1) - value_out_v.detach()

        log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
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
