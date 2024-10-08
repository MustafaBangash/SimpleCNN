# neural_network.py

import numpy as np
from .losses import (
    mse, cross_entropy,
    mse_derivative, cross_entropy_derivative
)
from .optimizers import SGD, Momentum, Adam

class NeuralNetwork:
    def __init__(self, loss='mse', optimizer='sgd', learning_rate=0.01):
        self.layers = []
        self.loss_function = self.get_loss_function(loss)
        self.loss_derivative = self.get_loss_derivative(loss)
        self.optimizer = self.get_optimizer(optimizer, learning_rate)

    def addLayer(self, layer):
        self.layers.append(layer)

    def get_loss_function(self, name):
        losses = {
            'mse': mse,
            'cross_entropy': cross_entropy
        }
        if name not in losses:
            raise ValueError(f"Unsupported loss function: {name}")
        return losses[name]

    def get_loss_derivative(self, name):
        derivatives = {
            'mse': mse_derivative,
            'cross_entropy': cross_entropy_derivative
        }
        if name not in derivatives:
            raise ValueError(f"Unsupported loss derivative function: {name}")
        return derivatives[name]

    def get_optimizer(self, name, learning_rate):
        optimizers = {
            'sgd': SGD(learning_rate),
            'momentum': Momentum(learning_rate),
            'adam': Adam(learning_rate)
        }
        if name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {name}")
        return optimizers[name]

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, batch_size):
        n_samples = x_train.shape[0]
        for epoch in range(epochs):
            err = 0
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
            for i in range(0, n_samples, batch_size):
                # Get batch data
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                err += self.loss_function(y_batch, output)

                # Backward pass
                error = self.loss_derivative(y_batch, output)
                for idx, layer in reversed(list(enumerate(self.layers))):
                    error = layer.backward(error, self.optimizer, idx)

            # Average error over batches
            err /= n_samples / batch_size
            print(f'Epoch {epoch +1}/{epochs} - Error: {err:.4f}')
