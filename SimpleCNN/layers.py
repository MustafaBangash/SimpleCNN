# layers.py

import numpy as np
from .activations import (
    sigmoid, relu, tanh,
    sigmoid_derivative, relu_derivative, tanh_derivative
)

class Layer:
    def __init__(self, n_input, n_neurons, activation='relu'):
        # Initialize weights and biases
        self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2. / n_input)
        self.biases = np.zeros((1, n_neurons))
        # Set activation function
        self.activation = activation
        self.activation_func = self.get_activation_function(activation)
        self.activation_derivative = self.get_activation_derivative(activation)
        # Placeholders for forward and backward pass
        self.input = None
        self.output = None
        self.z = None  # Linear combination output

    def get_activation_function(self, name):
        activations = {
            'relu': relu,
            'sigmoid': sigmoid,
            'tanh': tanh
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation function: {name}")
        return activations[name]

    def get_activation_derivative(self, name):
        derivatives = {
            'relu': relu_derivative,
            'sigmoid': sigmoid_derivative,
            'tanh': tanh_derivative
        }
        if name not in derivatives:
            raise ValueError(f"Unsupported activation function derivative: {name}")
        return derivatives[name]

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation_func(self.z)
        return self.output

    def backward(self, output_error, optimizer, layer_index):
        activation_error = self.activation_derivative(self.z) * output_error
        input_error = np.dot(activation_error, self.weights.T)
        weights_gradient = np.dot(self.input.T, activation_error)

        # Update weights and biases using the optimizer
        key_w = f'w_{layer_index}'
        key_b = f'b_{layer_index}'
        self.weights = optimizer.update(self.weights, weights_gradient, key_w)
        self.biases = optimizer.update(self.biases, activation_error.mean(axis=0, keepdims=True), key_b)

        return input_error
