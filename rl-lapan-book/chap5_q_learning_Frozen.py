import gym
import collections
import torch
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
INITIAL_STEPS = 100
TEST_EPISODES = 20
STOP_CRITERIA = 0.8

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.q_values = collections.defaultdict(float)

    def play_n_random_steps(self, n):
        s = self.state
        for i in range(n):
            a = self.env.action_space.sample()
            s_new, r, terminal, _ = self.env.step(a)
            # print("s,a,r,s_new : %d, %d, %.3f, %d" % (s, a, r, s_new))
            self.rewards[(s, a, s_new)] = r
            self.transits[(s, a)][s_new] += 1
            if terminal:
                s = self.env.reset()
            else:
                s = s_new

    def select_best_action(self, s):
        best_action, best_val = None, -99999
        for a in range(self.env.action_space.n):
            a_val = self.q_values[(s, a)]
            if a_val > best_val:
                best_action, best_val = a, a_val
        return best_action

    def play_one_episode(self, env):
        Ret = 0.0
        s = env.reset()
        while 1:
            a = self.select_best_action(s)
            s_new, r, terminal, _ = env.step(a)
            self.rewards[(s, a, s_new)] = r
            self.transits[(s,a)][s_new] += 1
            Ret += r
            if terminal:
                break
            s = s_new
            # print("s,a,r,s_new : %d, %d, %.3f, %d" % (s, a, r, s_new))
        return Ret

    def q_value_iteration(self):
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                action_value = 0.0
                d_targ2counts = self.transits[(s,a)]
                total_counts = sum(d_targ2counts.values())
                for s_new,v in d_targ2counts.items():
                    best_a_at_s_new = self.select_best_action(s_new)
                    action_value += (v*1.0/total_counts) * (self.rewards[(s,a, s_new)]
                        +GAMMA*self.q_values[(s_new, best_a_at_s_new)])
                self.q_values[(s, a)] = action_value


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-v-learning')

    i = 0
    best_Ret = 0.0
    while True:
        i += 1
        agent.play_n_random_steps(n=100)
        agent.q_value_iteration()
        # print(agent.q_values)
        # print(agent.rewards)

        Ret = 0.0
        for _ in range(TEST_EPISODES):
            Ret += agent.play_one_episode(test_env)
        avg_Ret = Ret / TEST_EPISODES
        writer.add_scalar("return", avg_Ret, i)

        print("%d-th iteration .... avg_Ret: %.3f" % (i, avg_Ret), end="")

        if avg_Ret > best_Ret:
            print("  | best_Ret updated [%.3f -> %.3f]" % (best_Ret, avg_Ret))
            best_Ret = avg_Ret
        else:
            print("  |")
        if avg_Ret > STOP_CRITERIA or i >= 100:
            break

    writer.close()