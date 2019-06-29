import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import random

batch_size = 64
input_size = 1000
output_size = 10
hidden_size = 100
LR = 0.01

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.linear1(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu)
        return y_pred

x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)

model = Net(input_size, hidden_size, output_size)
loss_fn = nn.MSELoss()
opt = opt.SGD(model.parameters(), lr=LR,  momentum=0.9)

for step_t in range(500):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    print(step_t, loss.item())

    model.zero_grad()
    loss.backward()

    opt.step()