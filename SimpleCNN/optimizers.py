# optimizers.py

import numpy as np

class Optimizer:
    def update(self, weights, gradients, key):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, weights, gradients, key=None):
        return weights - self.lr * gradients

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, weights, gradients, key):
        if key not in self.velocity:
            self.velocity[key] = np.zeros_like(gradients)
        self.velocity[key] = self.momentum * self.velocity[key] - self.lr * gradients
        return weights + self.velocity[key]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, weights, gradients, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(gradients)
            self.v[key] = np.zeros_like(gradients)
        self.t += 1
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients ** 2)
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
