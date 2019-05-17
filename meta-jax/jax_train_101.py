# based on https://github.com/google/jax#quickstart-colab-in-the-cloud

from jax import grad, jit
import jax.numpy as np

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.0) + 1)

def logistic_pred(w, x):
    return sigmoid(np.dot(x, w))

def loss(w, x, y):
    preds = logistic_pred(w, x)
    label_logprobs = np.log(preds) * y + np.log(1 - preds) * (1 - y)
    return -1.0 * np.sum(label_logprobs)

inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, False])

# define a complied function that returns gradients of the training loss
training_gradient_func = jit(grad(loss))

# GD opt
w = np.array([0.0, 0.0, 0.0]) # init
print("Initial loss: {:0.2f}".format(loss(w, inputs, targets)))
for i in range(100):
    print("%d-th iteration | loss: %.2f" % (i, loss(w, inputs, targets)) )
    w = w - 0.1 * training_gradient_func(w, inputs, targets)

print("Initial loss: {:0.2f}".format(loss(w, inputs, targets)))
