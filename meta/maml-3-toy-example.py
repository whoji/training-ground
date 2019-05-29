# Partially based on
# https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0
# full notebook : https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
# Author: Adrien Lucas Ecoffet

import numpy as np
import matplotlib.pyplot as plt
import pdb

LR = 0.01

class SineWaveTask:
    def __init__(self):
        self.a = np.random.uniform(0.1, 5.0)
        self.b = np.random.uniform(0, 2*np.pi)
        self.train_x = None
        self.train_y = None

    def f(self, x):
        return self.a * np.sin(x + self.b)

    # def training_set(self, size=10, force_new=False):
    #     if self.train_x is None and not force_new:
    #         self.train_x = np.random.uniform(-5, 5, size)
    #         x = self.train_x
    #     elif not force_new:
    #         x = self.train_x
    #     else:
    #         x = np.random.uniform(-5, 5, size)
    #     y = self.f(x)
    #     # return torch.Tensor(x), torch.Tensor(y)
    #     return x, y

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

    def plot(self, *args, **kwargs):
        x, y = self.test_set(size=100)
        return plt.plot(x, y, *args, **kwargs)

    def plot(self):
        x, y = self.test_set(size=100)
        plt.plot(x, y)
        x, y = self.training_set(size=10)
        plt.plot(x, y, '^')


# SineWaveTask().plot()
# SineWaveTask().plot()
# SineWaveTask().plot()
# plt.show()


# pdb.set_trace()

# for _ in range(50):
#     SineWaveTask().plot(color='black')
# plt.show()


####

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

    # def df(self, x):
    #     assert len(x) == self.degree
    #     ret = 0.0
    #     for i in range(self.degree):
    #         ret += (self.degree-i) * self.w[i]  *  (x[i] ** (self.degree-i-1))

    # go read this
    # https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html

    def grad_loss(self, x, y):
        ret = np.zeros(self.degree+1)
        f = [self.f(xj) for xj in x]
        for i in range(self.degree+1):
            for j in range(len(x)):
                ret[i] = 2 * (f[j] - y[j]) * x[j] ** (self.degree - i)
            ret[i] = 1 * ret[i] / len(x)
        return ret

    def loss(self, x, y):
        ret = 0.0
        f = [self.f(xj) for xj in x]
        for j in range(len(y)):
            ret += (f[j] - y[j])**2
        return ret / len(y)

    def update_model(self, x, y, lr=LR):
        grad = self.grad_loss(x, y)
        print(grad)
        print(self.loss(x, y))
        for i in range(self.degree+1):
            self.w[i] += lr * grad[i]

    def plot(self, x, y):
        x_test = np.linspace(-5, 5, 100)
        y_test = [self.f(xx) for xx in x_test]
        plt.plot(x_test, y_test)
        plt.plot(x, y, '^')



sw0 = SineWaveTask()
sw0.plot()

m0 = PolyFitModel(2)


# m0.grad_loss(list(sw0.train_x), list(sw0.train_y))


x = list(sw0.train_x)
y = list(sw0.train_y)
m0.loss(x, y)
m0.grad_loss(x, y)

'''
m0.update_model(x,y); print(m0); m0.plot(x,y); plt.show()
'''


pdb.set_trace()




