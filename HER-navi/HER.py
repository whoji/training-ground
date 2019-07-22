import numpy as np
from collections import deque
# list-like container with fast appends and pops on either end

class HER:
    '''
    the element in the HER is  [s, a, r, s_new, terminal]
    '''
    def __init__(self):
        self.buffer = deque()

    def reset(self):
        self.buffer = deque()

    def add(self, item):
        self.buffer.append(item)

    def backward(self):
        num = len(self.buffer)
        goal = self.buffer[-1][-2][1,:,:]

        for i in range(num):
            self.buffer[-1-i][-2][2,:,:] = goal
            self.buffer[-1-i][ 0][2,:,:] = goal
            self.buffer[-1-i][2] = -1.0
            self.buffer[-1-i][4] = False
            if np.sum(np.abs(self.buffer[-1-i][-2][1,:,:] - goal)) == 0:
                self.buffer[-1-i][2] = 0.0
                self.buffer[-1-i][4] = True

        return self.buffer