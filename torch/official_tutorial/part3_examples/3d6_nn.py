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

x = torch.randn(batch_size, input_size)
y = torch.randn(batch_size, output_size)

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

loss_fn = nn.MSELoss()

for step_t in range(500):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    print(step_t, loss.item())

    model.zero_grad()
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= LR * param.grad