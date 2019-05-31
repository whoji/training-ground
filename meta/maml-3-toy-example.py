# Partially based on
# https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0
# full notebook : https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
# Author: Adrien Lucas Ecoffet

import numpy as np
import matplotlib.pyplot as plt
import pdb

LR =  0.01

class SineWaveTask:
    def __init__(self):
        self.a = np.random.uniform(0.1, 5.0)
        self.b = np.random.uniform(0, 2*np.pi)
        self.train_x = None
        self.train_y = None

    def f(self, x):
        # return self.a * x + self.b + np.random.uniform(0.1, 5.0, len(x))
        return self.a * np.sin(0.7*x + self.b)

    def training_set(self, size=10):
        self.train_x = np.random.uniform(-5, 5, size)
        self.train_y = self.f(self.train_x)
        x = self.train_x
        y = self.f(x)
        return x, y

    def test_set(self, size=50):
        x = np.linspace(-5, 5, size)
        y = self.f(x)
        # return torch.Tensor(x), torch.Tensor(y)
        return x, y

    def plot(self):
        x, y = self.test_set(size=100)
        plt.plot(x, y)
        x, y = self.training_set(size=100)
        plt.plot(x, y, '^')


class PolyFitModel:
    def __init__(self, degree, lr=0.01):
        self.degree = degree
        self.w = np.random.uniform(-1, 1, degree+1)
        self.lr = lr

    def __repr__(self):
        #return str(self.w)
        ret = ""
        for i in range(self.degree):
            ret += "%.3fx^%d + " % (self.w[i] , self.degree-i)
        return ret + "%.3f" % self.w[-1]

    def f(self, x):
        ret = 0.0
        for i in range(self.degree + 1):
            ret += self.w[i] *  (x ** (self.degree-i))
        return ret

    def grad_loss(self, x, y):
        ret = np.zeros(self.degree+1)
        f = [self.f(xj) for xj in x]
        for i in range(self.degree+1):
            for j in range(len(x)):
                ret[i] += (- 2 * (y[j] - f[j]) * x[j] ** (self.degree - i) )
            ret[i] = ret[i] / len(x)
        return ret

    def loss(self, x, y):
        ret = 0.0
        f = [self.f(xj) for xj in x]
        for j in range(len(y)):
            ret += (f[j] - y[j])**2
        return ret / len(y)

    def update_model(self, x, y, lr=LR):
        #print("%s  |  loss: %.3f" % (self, self.loss(x, y)))
        grad = self.grad_loss(x, y)
        # print(self)
        #print("grad: "+str(grad))
        for i in range(self.degree+1):
            self.w[i] -= lr * grad[i] / len(x)

    def plot(self, x, y, c = 'blue'):
        x_test = np.linspace(-5, 5, 100)
        y_test = [self.f(xx) for xx in x_test]
        plt.plot(x_test, y_test, color=c)
        plt.plot(x, y, '^')


class MAML(object):
    """docstring for MAML"""
    def __init__(self, n_sets, n_shots, lr_a, lr_b):
        self.n_sets = n_sets
        self.n_shots = n_shots
        self.meta_train_set = self.get_meta_set(n_sets)
        self.meta_test_set = self.get_meta_set(n_sets)
        self.meta_model = PolyFitModel(lr =lr_b)
        self.lr_a, self.lr_b = lr_a, lr_b

    def get_meta_set(self, n):
        for i in range(n):
            task = SineWaveTask()
            append(task)

    def train(self):
        for task in self.meta_train_set:
            poly_model_i = self.meta_model.clone()
            ploy_model_i
            # TODO
            # WIP
            raise NotImplementedError

    def clone(self):
        raise NotImplementedError



def test_ploy_fit():
    LR = 0.02
    sw0 = SineWaveTask()
    sw0.plot()
    m0 = PolyFitModel(3)

    x = list(sw0.train_x)
    y = list(sw0.train_y)

    print("%d |  %s  |  loss: %.3f" % (0, m0, m0.loss(x, y)))
    print(m0.grad_loss(x, y))
    # print(m0); m0.plot(x,y,'yellow'); plt.show()

    m0.plot(x,y,'g')
    last_loss = 0
    for i in range(10):
        m0.update_model(x,y, LR); m0.plot(x,y,'m');
        cur_loss = m0.loss(x, y)
        if i % 100 ==0 : print("%d |  %s  |  loss: %.3f" % (i, m0, cur_loss))
        if abs((cur_loss - last_loss)/1) < 0.00001:
            break
        last_loss = cur_loss
    m0.update_model(x,y);  m0.plot(x,y,'r')
    axes = plt.gca(); axes.set_ylim([-5,5])

    plt.show()

# pdb.set_trace()




