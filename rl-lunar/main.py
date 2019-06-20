# https://www.youtube.com/watch?v=UlJzzLYgYoE
# remember to install pybox2d: pip install box2d-py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np

import gym

class Net(nn.Module):
    def __init__(self, lr, input_size, hid_1, hid_2, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(*input_size, hid_1)
        self.fc2 = nn.Linear(hid_1, hid_2)
        self.fc3 = nn.Linear(hid_2, n_actions)
        self.opt = opt.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.device = T.device('cuda:0' if torch.cuda.is_available() )
        self.device = torch.device('cuda:0')
        self.to(self.device)

    def forward(self, s):
        s_v =torch.Tensor(s).to(self.device)
        x = F.relu(self.fc1(s_v))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_size, batch_size,
        n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=0.996):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.max_mem_size = max_mem_size

        self.mem_cntr = 0
        self.q_eval = Net(lr, n_actions=self.n_actions,
            input_size=input_size, hid_1=256, hid_2=256)
        self.state_memory = np.zeros((self.max_mem_size, *input_size))
        self.new_state_memory = np.zeros((self.max_mem_size, *input_size))
        self.action_memory = np.zeros((self.max_mem_size, self.n_actions),
            dtype=np.uint8)
        self.reward_memory = np.zeros(self.max_mem_size)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.uint8)

    def store_transition(self, s, a, r, s_new, terminal):
        idx = self.mem_cntr % self.max_mem_size
        self.state_memory[idx] = s
        actions = np.zeros(self.n_actions)
        actions[a]  = 1.0
        self.action_memory[idx] = actions
        self.reward_memory[idx] = r
        self.terminal_memory[idx]   = terminal
        self.new_state_memory[idx] = s_new
        self.mem_cntr += 1

    def choose_action(self, s):
        rand = np.random.random()
        if rand < self.epsilon:
            a = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.forward(s)
            a = torch.argmax(actions).item()
        return a

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.q_eval.opt.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.max_mem_size else self.max_mem_size
            batch_idx = np.random.choice(max_mem, self.batch_size)
            s_batch = self.state_memory[batch_idx]
            a_batch = self.action_memory[batch_idx]
            a_values = np.array(self.action_space, dtype=np.uint8)
            a_idx =np.dot(a_batch, a_values)

            # print(a_batch)
            # print(a_values)
            # print(a_idx)
            # print("-------------------")

            r_batch = self.reward_memory[batch_idx]
            terminal_batch = self.terminal_memory[batch_idx]
            s_new_batch = self.new_state_memory[batch_idx]

            r_batch = torch.Tensor(r_batch).to(self.q_eval.device)
            terminal_batch = torch.Tensor(terminal_batch).to(self.q_eval.device)

            q_eval = self.q_eval.forward(s_batch).to(self.q_eval.device)
            q_tgt = self.q_eval.forward(s_batch).to(self.q_eval.device)
            q_next = self.q_eval.forward(s_new_batch).to(self.q_eval.device)

            batch_idx = np.arange(self.batch_size, dtype=np.int32)


            # print(q_tgt.shape)
            # print(batch_idx.shape)
            # print(batch_idx)
            # print(a_idx.shape)

            # print(q_tgt[:,1])
            # print(q_tgt[:,(1,2)])

            # import pdb; pdb.set_trace()


            #print(q_tgt[a_batch])

            #import pdb; pdb.set_trace()


            #q_tgt[batch_idx, a_idx] = \
            q_tgt[a_batch] = \
                r_batch + self.gamma * torch.max(q_next, dim=1)[0]*terminal_batch

            self.epsilon = self.epsilon*self.eps_dec if self.epsilon \
                > self.eps_end else self.eps_end

            loss = self.q_eval.loss(q_tgt, q_eval).to(self.q_eval.device)
            loss.backward()
            self.q_eval.opt.step()

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
        input_size = [8], lr=0.003)
    scores = []
    eps_history = []
    n_games = 500
    score = 0

    for i in range(n_games):
        if i % 10 ==0 and i > 0:
            avg_score = np.mean(scores[-10:])
            print('%i %.3f %.3f %.3f' % (i, agent.epsilon, score, avg_score))

        score = 0
        eps_history.append(agent.epsilon)
        s = env.reset()
        terminal = False
        while not terminal:
            a = agent.choose_action(s)
            s_new, r, terminal, _  = env.step(a)
            score +=r
            agent.store_transition(s,a,r,s_new,terminal)
            agent.learn()
            s = s_new
        scores.append(score)

    x = [i+1 for i in range(n_games)]


