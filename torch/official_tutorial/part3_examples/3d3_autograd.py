import torch

dtype = torch.float
device = torch.device("cpu")

batch_size = 64
input_size = 1000
output_size = 10
hidden_size = 100
LR = 1e-6

x = torch.randn(batch_size, input_size, dtype= dtype)
y = torch.randn(batch_size, output_size, dtype= dtype)

w1 = torch.randn(input_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(hidden_size, output_size, device=device, dtype=dtype, requires_grad=True)

for step in range(500):
    h1 = x.mm(w1).clamp(min=0)
    y_pred = h1.mm(w2)
    # y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(step, loss.item())

    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        # w1 = w1 - LR * w1.grad  # fuck ! this is not same as w1 -= LR * w1.grad
        # w2 = w2 - LR * w2.grad  # Wrong wrong wrong
        w1 -= LR * w1.grad
        w2 -= LR * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()