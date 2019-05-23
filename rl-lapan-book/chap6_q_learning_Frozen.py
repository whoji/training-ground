# modified from chap5_q_learning_Frozen.py

import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20
STOP_CRITERIA = 0.8

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # self.rewards = collections.defaultdict(float)
        # self.transits = collections.defaultdict(collections.Counter)
        self.q_values = collections.defaultdict(float)

    def sample_env(self):
        s = self.state
        a = self.env.action_space.sample()
        s_new, r, terminal, _ = self.env.step(a)
        # print("s,a,r,s_new : %d, %d, %.3f, %d" % (s, a, r, s_new))
        if terminal:
            self.state = self.env.reset()
        else:
            self.state = s_new
        return s, a, r, s_new

    def get_best_value_and_action(self, s):
        best_action, best_val = None, -99999
        for a in range(self.env.action_space.n):
            a_val = self.q_values[(s, a)]
            if a_val > best_val:
                best_action, best_val = a, a_val
        return best_val, best_action

    def q_value_update(self, s, a, r, s_new):
        best_v_for_s_new, _ = self.get_best_value_and_action(s_new)
        new_val = r + GAMMA * best_v_for_s_new
        old_val = self.q_values[(s,a)]
        self.q_values[(s,a)] = (1-ALPHA) * old_val + ALPHA * new_val

    def play_one_episode(self, env):
        Ret = 0.0
        s = env.reset()
        while 1:
            _, a = self.get_best_value_and_action(s)
            s_new, r, terminal, _ = env.step(a)
            Ret += r
            if terminal:
                break
            s = s_new
            # print("s,a,r,s_new : %d, %d, %.3f, %d" % (s, a, r, s_new))
        return Ret


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-v-learning')

    i = 0
    best_Ret = 0.0
    while True:
        i += 1
        s, a, r, s_new = agent.sample_env()
        agent.q_value_update(s, a, r, s_new)
        # print(agent.q_values)
        # print(agent.rewards)

        Ret = 0.0
        for _ in range(TEST_EPISODES):
            Ret += agent.play_one_episode(test_env)
        avg_Ret = Ret / TEST_EPISODES
        writer.add_scalar("return", avg_Ret, i)

        # print("%d-th iteration .... avg_Ret: %.3f" % (i, avg_Ret), end="")
        if avg_Ret > best_Ret:
            print("  | best_Ret updated [%.3f -> %.3f]" % (best_Ret, avg_Ret))
            best_Ret = avg_Ret
        # else:
        #     print("  |")
        if avg_Ret > STOP_CRITERIA:
            print("solved in %d iteractions ..." % i)
            break
        if  i >= 20000:
            print("not solved in %d iteractions ..." % i)
            break

    writer.close()