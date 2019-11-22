import torch
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
print(a, b)

lr = 1e-1
n_epochs = 1000

# Defines a SGD optimizer to update the parameters
optimizer = optim.SGD([a, b], lr=lr)

loss_fn = nn.MSELoss(reduction='mean')

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor

    # No more manual loss!
    # error = y_tensor - yhat
    # loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(a, b)