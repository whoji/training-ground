import numpy as np

N = 64          # batch size
D_in = 1000     # input size
H = 100         # hidden size
D_out = 10      # output size
LR = 1e-6       # learning rate

# create random dataset for training (input and output)
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# init weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0) # element wise max. if np.max(), return 1 val
    y_pred = h_relu.dot(w2)

    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # backprop to compute grad of w1 and w2, wrt loss

    # by chain rule dloss/dw2 = dloss/dy_pred * dy_pred/d_w2
    #                         = 2*(y_pred-y)  * h_relu
    d_y_pred = 2.0 * (y_pred - y)
    g_w2 = h_relu.T.dot(d_y_pred)

    # by chain rule dloss/dw1 = dloss/dy_pred * dy_pred/d_w1
    #                         = dloss/dy_pred * w2 * dh_relu/d_w1
    #                         = dloss/dy_pred * w2 * [x (or 0 if h < 0)]
    d_h_relu = d_y_pred.dot(w2.T)
    d_h = d_h_relu.copy()
    d_h[h < 0] = 0
    g_w1 = x.T.dot(d_h)

    w1 = w1 - LR * g_w1
    w2 = w2 - LR * g_w2