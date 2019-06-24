import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
import gym
import pybullet_envs
import time

GAMMA = 0.99
EpisodeStep = namedtuple('EpisodeStep', field_names=['s', 'a', 'r'])

# this is taken from ptan's agent class
# https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class AgentA2C():
    def __init__(self, net, device='cpu'):
        self.net = net
        self.device = device

    def predict(self, states):
        s_v = float32_preprocessor(states).to(self.device)
        mu_v, var_v, _ = self.net(s_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions

    def iterate_sample(self, env, steps):
        episode_steps = []
        s = env.reset()
        # sm = nn.Softmax(dim=1)

        while True:
            actions = self.predict([s])
            a = actions[0]
            # alternative approach: just take the mean. no exploration
            # a = mu_v.squeeze(dim=0).data.cpu().numpy()
            # a = np.clip(a, -1, 1)

            s_new, r, terminal, _ = env.step(a)
            episode_steps.append(EpisodeStep(s=s,a=a,r=r))

            if terminal:
                S, A, R = process_episode(episode_steps, steps = steps)
                for i in range(len(S)):
                    Ret = len(S) if i == len(S)-1  else None
                    # pdb.set_trace()
                    yield (S[i], A[i], R[i], Ret)
                episode_steps = []
                s_new = env.reset()

            s = s_new

    def watch_with_render_continous(self, env_name, episodes=1, horizon=1000):
        # import pdb; pdb.set_trace()
        spec = gym.envs.registry.spec(env_name)
        spec._kwargs['render'] = True

        for ep in range(episodes):
            env = gym.make(env_name)
            s = env.reset()
            frames = 0
            Ret = 0.0
            for _ in range(horizon):
                actions = self.predict([s])
                a = actions[0]
                s_new, r, terminal, _ = env.step(a)
                Ret += r
                if terminal:
                    print("finished %d/%d episode !! Frames=%d Ret=%.3f" % (
                        ep+1, episodes, frames, Ret))
                    frames = 0
                    Ret = 0.0
                    time.sleep(5)
                    break
                else:
                    frames += 1
                    s = s_new
        env.close()

def process_episode(episode_steps, steps):
    s = [e.s for e in episode_steps]
    a = [e.a for e in episode_steps]
    r = [e.r for e in episode_steps]
    n = len(episode_steps)
    q = []
    for i in range(n):
        # import pdb; pdb.set_trace()
        r_roll = 0.0
        for j in range(steps):
            if i+j >= n:
                break
            r_roll += r[i+j] * GAMMA**j
        q.append(r_roll)
    assert len(q) == len(r)
    # import pdb; pdb.set_trace()
    return (s, a, q)


