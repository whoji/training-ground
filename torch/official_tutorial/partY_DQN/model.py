import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition',('s', 'a', 's_new', 'r'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity    # max_size
        self.memory = []            # the replay memory itself
        self.position = 0           # cursor position

    def push(self, *args):
        """Saves a transition.
        even if the size exceeds capacity, you can still push by
        replacing the oldest ones
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
instead of acquiring the internal states/observations, let's do the
CV approach. We’ll use a patch of the screen centered on the cart
as an input states (observations).
'''
class DQN(nn.Module):
    def __init__(self, h, w, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, 2)
        self.bn3   = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output
        # of conv2d layers (conv3 and bn3)
        # and therefore the input image size, so compute it.
        def conv2d_output_size(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size-1)-1) // stride + 1
        conv1_o_w = conv2d_output_size(w)
        conv1_o_h = conv2d_output_size(h)
        conv2_o_w = conv2d_output_size(conv1_o_w)
        conv2_o_h = conv2d_output_size(conv1_o_h)
        conv3_o_w = conv2d_output_size(conv2_o_w)
        conv3_o_h = conv2d_output_size(conv2_o_h)
        fc_input_size = conv3_o_h * conv3_o_w * 32

        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # flatten
        return self.fc(x)