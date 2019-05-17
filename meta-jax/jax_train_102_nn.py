# from https://blog.evjang.com/2019/02/maml-jax.html

from jax import vmap # for auto-vectorizing functions
from functools import partial # for use with vmap
from jax import jit # for compiling functions for speedup
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax # neural network layers
import matplotlib.pyplot as plt # visualization

net_init, net_apply = stax.serial(
    Dense(40), Relu,
    Dense(40), Relu,
    Dense(1)
)

in_shape = (-1, 1)

out_shape, net_params = net_init(rng = 0, input_shape = in_shape)

def loss(w, x, y):
    # average loss of the batch (MSE)
    y_hat = net_apply(w, x)
    return np.mean((y - y_hat)**2)

# batch the inference across K = 100
x = np.linspace(-5,5,100).reshape((100,1)) # (k, 1)
y = np.sin(x)
y_hat = vmap(partial(net_apply, net_params))(x)
losses = vmap(partial(loss, net_params))(x, y)
plt.plot(x, y_hat, label='y_hat: predictions')
plt.plot(x, losses, label='loss')
plt.plot(x, y, label='y: truth')
plt.legend()