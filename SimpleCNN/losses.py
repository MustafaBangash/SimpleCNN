# losses.py

import numpy as np

# Loss Functions
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def cross_entropy(y_true, y_pred):
    epsilon = 1e-12  # To prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Loss Function Derivatives
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true
