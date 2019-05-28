import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """docstring for DQN"""
    def __init__(self, input_shape, n_actions):
        """assumes input_shape is of CHW shape (4,84,84)"""
        super(DQN, self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        temp_conv_out = self.conv(torch.zeros(1, *shape))
        # import pdb; pdb.set_trace()
        return int(np.prod(temp_conv_out.size())) # 1*128*7*7 = 6272

    def forward(self, x):
        """ assumes x is 4D of shape BCHW, output will be 2D: B*n_actions """
        conv_out = self.conv(x).view(x.size()[0], -1) # flatten ? what this is not 1D ??!!
        return self.fc(conv_out)


if __name__ == '__main__':
    m = DQN((4,100,100), 5)
    print(m)
