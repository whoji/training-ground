# based on https://jhui.github.io/2018/02/09/PyTorch-neural-networks/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net1 = Net()
print(net1)


### print out learnables
print("="*100)
params = list(net1.parameters())
print(len(params))

for i in range(10):
    print(i, params[i].size())

print("="*100)

## test
input = Variable(torch.randn(1, 1, 32, 32))
print(input)
out = net1(input)   #

print(out)


## =================
print("="*100)


#net1.zero_grad()
#out.backward()

target = Variable(torch.arange(1,11).float())
criterion = nn.MSELoss()

loss = criterion(out, target)
net1.zero_grad()
loss.backward()

print(net1.conv1.bias.grad)

 ### -------------
opt = opt.SGD(net1.parameters(), lr=0.01)

for t in range(500):
    output = net1(input)
    loss = criterion(output, target)

    if t % 10 == 0:
        print(t, loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()



