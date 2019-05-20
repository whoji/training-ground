# based on the chap4 of Maxim Lapan

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128	 # width of the NN
BATCH_SIZE = 32      # 32 episode in one batch
TOP_N_PERCENT = 0.3   # top 30% if tge eouside fir training
LEARNING_RATE = 0.01

Episode = namedtuple('Episode', field_names=['return_','steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['s', 'a', 'r'])

class Net(nn.Module):
	def __init__(self, obs_size, hidden_size, n_action):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_action)
		)
		
	def forward(self, x):
		return self.net(x)


def iterate_batches(env, net, batch_size):
	batch = []
	episode_reward = 0.0
	episode_steps = []
	s = env.reset()
	sm = nn.Softmax(dim=1)

	while True:
		obs_v = torch.FloatTensor([s])
		nn_output_v = net(obs_v)
		act_probs_v = sm(nn_output_v)
		act_probs = act_probs_v.data.numpy()[0]
		a = np.random.choice(len(act_probs), p=act_probs)
		s_new, r, terminal, _ = env.step(a)

		episode_reward += r
		episode_steps.append(EpisodeStep(s=s,a=a,r=r))
		if terminal:
			batch.append(Episode(return_=episode_reward, steps=episode_steps))
			episode_reward = 0.0
			episode_steps = []
			s_new = env.reset()
			if len(batch) == batch_size:
				yield batch
				batch = []

		s = s_new

def filter_batch(batch, percent=TOP_N_PERCENT):
	returns = list(map(lambda s: s.return_, batch))
	returns_sorted = sorted(returns, reverse = True)
	return_cutoff = returns_sorted[round(len(returns_sorted)*percent)]
	return_mean = np.mean(returns)

	train_s = []
	train_a = []
	for ep in batch:
		if ep.return_ >= return_cutoff:
			train_s = train_s + [step.s for step in ep.steps]
			train_a = train_a + [step.a for step in ep.steps]

	train_s_v = torch.FloatTensor(train_s)
	train_a_v = torch.LongTensor(train_a)
	return train_s_v, train_a_v, return_cutoff, return_mean


if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	# env = gym.wrappers.Monitor(env, directory="mon", force=True)
	s_size = env.observation_space.shape[0]
	a_size = env.action_space.n

	net = Net(s_size, HIDDEN_SIZE, a_size)
	obj_func = nn.CrossEntropyLoss()
	opt = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
	writer = SummaryWriter()

	for i , batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
		train_s_v, train_a_v, return_cutoff, return_mean = \
		   filter_batch(batch, percent=TOP_N_PERCENT)
		opt.zero_grad()
		a_pred_v = net(train_s_v)
		loss_v = obj_func(a_pred_v, train_a_v)
		loss_v.backward()
		opt.step()

		print("%d: loss=%.3f, ret_mu=%.3f, ret_cut=%.3f" % (
			i, loss_v.item(), return_mean, return_cutoff))
		writer.add_scalar("loss", loss_v.item(), i)
		writer.add_scalar("ret_mu", return_mean, i)
		writer.add_scalar("ret_cut", return_cutoff, i)

		if return_mean > 199: 
			print("GGWP: Solved!")
			break

	writer.close()