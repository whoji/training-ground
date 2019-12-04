# started 11.26.2019
# ref 1 : https://github.com/pytorch/examples/blob/master/mnist/main.py
# ref 2 : https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetShared(nn.Module):

    def __init__(self):
        super(NetShared, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetMain(nn.Module):

    def __init__(self, shared_net):
        super(NetMain, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.shared_net = shared_net

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        output = self.shared_net(x)
        return output


device = "cpu"

N = 1000

x_1 = torch.randn(N, 1, 28, 28)
y_1 = torch.randint(0, 10, size=(N,))

x_2 = torch.randn(N, 1, 28, 28)
y_2 = torch.randint(0, 10, size=(N,))


shared_model = NetShared().to(device)
model_1 = NetMain(shared_model)
model_2 = NetMain(shared_model)

# model_1 = Net()


# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.01, momentum=0.9)
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01, momentum=0.9)
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred_1 = model_1(x_1)
    y_pred_2 = model_2(x_2)

    # Compute and print loss
    loss_1 = criterion(y_pred_1, y_1)
    loss_2 = criterion(y_pred_2, y_2)
    if t % 10 == 0:
        print(t, loss_1.item(), loss_2.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    loss_1.backward()
    loss_2.backward()
    optimizer_1.step()
    optimizer_2.step()
