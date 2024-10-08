# activations.py

import numpy as np

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Activation Function Derivatives
def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
