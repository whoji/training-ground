import gym
import random

class RandomActionWrapper(gym.ActionWrapper):
	"""docstring for RandomActionWrapper"""
	def __init__(self, env, epsilon=0.5):
		super(RandomActionWrapper, self).__init__(env)
		self.epsilon = epsilon

	def action(self, action):
		if random.random() < self.epsilon:
			a = self.env.action_space.sample()
			print("random action !! [%d --> %d]" % (action, a))
			return a 
		else:
			return action
		

if __name__ == '__main__':
	env = RandomActionWrapper(gym.make("CartPole-v0"))

	s = env.reset()
	total_return = 0.0

	while 1:
		s, r, terminal, _ = env.step(0)
		total_return += r
		if terminal:
			break

	print("Total return: %.3f" % total_return)
