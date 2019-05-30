import gym
import gym.spaces
import numpy as np
import collections
import cv2

MAX_LEN = 2
SKIP = 4

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, a):
        return self.env.step(a)

    # def reset(self):
    #     self.env.reset()
    #     s, _, terminal, _ = self.env.step(1)
    #     if terminal:
    #         self.env.reset()
    #     s, _, terminal, _ = self.env.step(2)
    #     if terminal:
    #         self.env.reset()
    #     return s

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    """docstring for MaxAndSkipEnv"""
    def __init__(self, env=None, skip=SKIP, maxlen=MAX_LEN):
        super(MaxAndSkipEnv, self).__init__(env)
        self._s_buffer = collections.deque(maxlen=maxlen)
        self._skip = skip

    def step(self, a):
        total_reward = 0.0
        terminal = None
        for _ in range(self._skip):
            s, r, terminal, info = self.env.step(a)
            self._s_buffer.append(s)
            total_reward += r
            if terminal:
                break
        max_frame = np.max(np.stack(self._s_buffer), axis = 0)
        return max_frame, total_reward, terminal, info

    def _reset(self):
        self._s_buffer.clear()
        s = self.env.reset()
        self._s_buffer.append(s)
        return s


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(84,84,1), dtype=np.uint8)

    def observation(self, s):
        return ProcessFrame84.process(s)

    @staticmethod
    def process(frame):
        if frame.size == 210*160*3:
            img = np.reshape(frame, [210,160,3]).astype(np.float32)
        elif frame.size == 250*160*3:
            img = np.reshape(frame, [250,160,3]).astype(np.float32)
        else:
            # raise Exception("Unknown resolution !!")
            assert False, "Unknown resolution !!"
        img = img[:,:,0] *0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84,84,1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        print(old_space)
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)
        print(self.observation_space)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=
            (old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32)/255.0


def make_env(env_name):
    # the following order MATTERS!
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env

if __name__ == '__main__':
    # test the wrapper
    env = make_env("PongNoFrameskip-v4")