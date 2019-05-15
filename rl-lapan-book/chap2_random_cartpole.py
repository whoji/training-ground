import gym

if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	# env = gym.wrappers.Monitor(env, "recording")
	total_reward = 0.0
	total_steps = 0
	obs = env.reset()

	while 1:
		action = env.action_space.sample()
		obs, rwd, done, _  = env.step(action)
		total_reward += rwd
		total_steps += 1
		print("step:%d, action: %d, rwd: %.4f" % (total_steps, action, rwd))
		if done:
			break

	print("Episode doen in %d steps, total reward %.2f" % (total_steps, total_reward))

