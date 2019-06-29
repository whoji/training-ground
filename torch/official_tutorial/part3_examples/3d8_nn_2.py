import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np

batch_size = 64
input_size = 1000
output_size = 10
hidden_size = 100
LR = 0.01

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_relue = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relue)
        return y_pred

x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)

model = Net(input_size, hidden_size, output_size)

#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = nn.MSELoss()

opt = opt.SGD(model.parameters(), lr=LR)

for step_t in range(500):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    print(step_t, loss.item())

    model.zero_grad()
    loss.backward()

    opt.step()