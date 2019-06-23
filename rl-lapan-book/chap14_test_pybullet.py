import gym
import pybullet_envs

ENV = "MinituarBulletEnv-v0"
RENDER = True

if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV)

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    print(env)
    print(env.reset())
    input("Press any key to exit")
    env.close()