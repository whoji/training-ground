import torch

dtype = torch.float
device = torch.device("cpu")

N = 64          # batch size
D_in = 1000     # input size
H = 100         # hidden size
D_out = 10      # output size
LR = 1e-6       # learning rate

# create random dataset for training (input and output)
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out,device=device, dtype=dtype)

# init weights
w1 = torch.randn(D_in, H,  device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

for t in range(500):
    h = x.mm(w1) #  mm: matrix multiplication
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # tensor.item() to get a Python number from a tensor containing
    # a single value:
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # backprop to compute grad of w1 and w2, wrt loss

    # by chain rule dloss/dw2 = dloss/dy_pred * dy_pred/d_w2
    #                         = 2*(y_pred-y)  * h_relu
    d_y_pred = 2.0 * (y_pred - y)
    g_w2 = h_relu.t().mm(d_y_pred)

    # by chain rule dloss/dw1 = dloss/dy_pred * dy_pred/d_w1
    #                         = dloss/dy_pred * w2 * dh_relu/d_w1
    #                         = dloss/dy_pred * w2 * [x (or 0 if h < 0)]
    d_h_relu = d_y_pred.mm(w2.t())
    d_h = d_h_relu.clone()
    d_h[h < 0] = 0
    g_w1 = x.t().mm(d_h)

    w1 = w1 - LR * g_w1
    w2 = w2 - LR * g_w2
