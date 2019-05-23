# based on the chap6 of Maxim Lapan
# code modified from chap4_CE_CartPole
# detailed maths. check this link
# http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_4_policy_gradient.pdf


import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ptan
from collections import namedtuple
from tensorboardX import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODE_TO_TRAIN = 4
ENV_NAME = "CartPole-v0"
HIDDEN_SIZE = 128	 # width of the NN
STOP_CRITERIA = 195

# BATCH_SIZE = 32      # 32 episode in one batch
# TOP_N_PERCENT = 0.3   # top 30% if tge eouside fir training


#Episode = namedtuple('Episode', field_names=['return_','steps'])
#EpisodeStep = namedtuple('EpisodeStep', field_names=['s', 'a', 'r'])

class PGNet(nn.Module):
	# rember the output of the net is not prob, but raw  scores.
	# need to use log_softmax func later.
	def __init__(self, obs_size, hidden_size, n_action):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_action)
		)

	def forward(self, x):
		return self.net(x)


def calc_qvals(rewards):
	# given the rewards trajectory, assigned discount total_reward(return)
	res = []
	sum_r = 0.0
	for r in reversed(rewards):
		sum_r *= GAMMA
		sum_r += r
		res.append(sum_r)
	return res[::-1]

if __name__ == '__main__':
	env = gym.make(ENV_NAME)
	writer = SummaryWriter(comment="-chap6pg1")

	net = PGNet(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n)
	print(net)

	agent = ptan.agent.PolicyAgent(net,
		preprocessor=ptan.agent.float32_preporcessor,apply_softmax=True)

	exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

	opt = optim.Adam(net.parameters(), lr=LEARNING_RATE)

	total_rewards = [] # return
	doen_episdoes = []

	batch_episodes = 0
	cur_rewards = []
	batch_s, batch_a, batch_qval = [], [], []

	for step_idx, exp in enumerate(exp_source):
		batch_s.append(exp.state)
		batch_a.append(int(exp.action))
		cur_rewards.append(exp.reward)

		if exp.last_state is None:
			batch_qval += calc_qvals(cur_rewards)
			cur_rewards.clear()
			batch_episodes += 1

		new_rewards = exp_source.pop_total_rewards()
		if new_rewards:
			doen_episdoes += 1
			r = new_rewards[0]
			total_rewards.append(r)
			mean_rewards = float(np.mean(total_rewards[-100:]))
			print("%d -th: [reward: %.3f] [mean: %.3f] [episodes: %d] " %
				(step_idx, r, mean_rewards, doen_episdoes))
			writer.add_scalar("reward", r, step_idx)
			writer.add_scalar("rwd_mean_100", mean_rewards, step_idx)
			writer.add_scalar("episodes", doen_episdoes, step_idx)
			if mean_rewards > STOP_CRITERIA:
				print("GG solved in %d steps and %d episodes" % (step_idx, doen_episdoes))
				break

		if batch_episodes < EPISODE_TO_TRAIN:
			continue

		## train the network
		opt.zero_grad()
		s_v = torch.FloatTensor(batch_s)
		a_t = torch.LongTensor(batch_a)
		q_v = torch.FloatTensor(batch_qval)

		net_output = net(s_v)
		log_prob_v = F.log_softmax(net_output, dim=1)
		log_prob_actions_v = q_v * log_prob_v[range(len(batch_s)), a_t]
		loss_v = - log_prob_actions_v.mean()

		loss_v.backward()
		opt.step()
		batch_episodes = 0
		batch_s = []
		batch_a = []
		batch_qval = []

	wrtier.close()


