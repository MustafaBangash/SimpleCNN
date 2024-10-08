# SimpleCNN: A Python Library for Convolutional Neural Networks

SimpleCNN is a Python library designed to make the development of Convolutional Neural Networks (CNNs) accessible and straightforward for AI beginners. Whether you are a student stepping into the world of AI, a researcher experimenting with CNN architectures, or a developer aiming to integrate CNNs into applications, SimpleCNN provides a user-friendly interface to build and train CNN models with ease.

## Features

- **Easy to Use**: SimpleCNN offers a high-level tool that abstracts away the complexities typically associated with building CNNs.
- **Flexible**: Configure your neural networks layer by layer, with support for custom configurations tailored to your specific datasets.
- **Educational**: Provides excellent support for learning, with clear documentation on each function and method to help beginners understand what's happening behind the scenes.

## Installation

You can install SimpleCNN directly from GitHub using pip:

```bash
pip install git+https://github.com/MustafaBangash/SimpleCNN.git
```
## Getting Started

Below is a guide to help you start using SimpleCNN.

### 1. Import the Library
After installation, import the necessary classes:

```
from SimpleCNN import NeuralNetwork, Layer
```

### 2. Create a Neural Network Instance
Create an instance of the NeuralNetwork class. You can specify the following parameters:

```
nn = NeuralNetwork(loss='mse', optimizer='sgd', learning_rate=0.01)
```
**Parameters:**
- loss (str): Loss function to use. Options are `'mse'` (Mean Squared Error) for regression tasks or `'cross_entropy'` for classification tasks. Default is `'mse'`.
- optimizer (str): Optimization algorithm. Options are `'sgd'` (Stochastic Gradient Descent), `'momentum',` or `'adam'`. Default is `'sgd'`.
- learning_rate (float): Learning rate for the optimizer. Default is `0.01`.

  
Example:

```
nn = NeuralNetwork(loss='cross_entropy', optimizer='adam', learning_rate=0.001)
```
### 3. Add Layers to the Network
Add layers to your network using the addLayer() method. Each layer is an instance of the Layer class.

```
nn.addLayer(Layer(n_input, n_neurons, activation='relu'))
```

**Parameters:**
- n_input (int): Number of input features to the layer (number of nodes in the previous layer).
- n_neurons (int): Number of neurons in the layer.
- activation (str): Activation function. Options are 'relu', 'sigmoid', or 'tanh'. Default is 'relu'.
- 
**Example:**

Add an input layer (for MNIST dataset with 28x28 images flattened to 784 features)
```
nn.addLayer(Layer(n_input=784, n_neurons=128, activation='relu'))
```
Add a hidden layer
```
nn.addLayer(Layer(n_input=128, n_neurons=64, activation='tanh'))
```
Add an output layer (for 10 classes)
```
nn.addLayer(Layer(n_input=64, n_neurons=10, activation='sigmoid'))
```
### 4. Train the Network
Train the network using the fit() method.

```
nn.fit(x_train, y_train, epochs=10, batch_size=32)
```
**Parameters:**
- x_train (numpy array): Training data features.
- y_train (numpy array): Training data labels.
- epochs (int): Number of times the entire training dataset passes through the network.
- batch_size (int): Number of samples per gradient update.
- 
Example: Assuming train_images and train_labels are prepared numpy arrays
```
nn.fit(x_train=train_images, y_train=train_labels, epochs=10, batch_size=32)
```
### 5. Make Predictions
Use the predict() method to make predictions on new data.

```
predictions = nn.predict(x_test)
```
**Parameters:**
- x_test (numpy array): Test data features.
- Example:

### 6. Evaluate the Network
Since the NeuralNetwork class does not have an evaluate method, you can manually calculate the accuracy.

Example:

```
import numpy as np

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
Full Example Code
```
import numpy as np
from SimpleCNN import NeuralNetwork, Layer

# Initialize the neural network with specified loss function and optimizer
nn = NeuralNetwork(loss='cross_entropy', optimizer='adam', learning_rate=0.001)

# Add layers to the network
nn.addLayer(Layer(n_input=784, n_neurons=128, activation='relu'))   # Input layer
nn.addLayer(Layer(n_input=128, n_neurons=64, activation='tanh'))    # Hidden layer
nn.addLayer(Layer(n_input=64, n_neurons=10, activation='sigmoid'))  # Output layer

# Prepare your data here
# train_images, train_labels = ... (Your training data)
# test_images, test_labels = ... (Your test data)

# Train the network
nn.fit(x_train=train_images, y_train=train_labels, epochs=10, batch_size=32)

# Make predictions on test data
predictions = nn.predict(test_images)

# Evaluate the network
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)
accuracy = np.mean(predicted_labels == true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
## Parameter Options Summary
### NeuralNetwork Class
```
class NeuralNetwork:
    def __init__(self, loss='mse', optimizer='sgd', learning_rate=0.01):
        # loss: 'mse' or 'cross_entropy'
        # optimizer: 'sgd', 'momentum', or 'adam'
        # learning_rate: positive float
```
**loss:**
- 'mse': Mean Squared Error (useful for regression tasks).
- 'cross_entropy': Cross-Entropy Loss (useful for classification tasks).
**optimizer:**
- 'sgd': Stochastic Gradient Descent.
- 'momentum': SGD with Momentum.
- 'adam': Adaptive Moment Estimation.
**learning_rate:**
- A positive float value that controls the step size during optimization.
  
### Layer Class
```
class Layer:
    def __init__(self, n_input, n_neurons, activation='relu'):
        # n_input: number of input features
        # n_neurons: number of neurons in the layer
        # activation: 'relu', 'sigmoid', or 'tanh'
```
**n_input:**
- The number of input features from the previous layer.
**n_neurons:**
- The number of neurons (nodes) in this layer.
**activation:**
- 'relu': Rectified Linear Unit activation function.
- 'sigmoid': Sigmoid activation function.
- 'tanh': Hyperbolic Tangent activation function.
- 
### Tips
Data Preprocessing: Ensure your input data is properly normalized and reshaped to match the expected input size.
One-Hot Encoding: For classification tasks, your labels should be one-hot encoded when using cross-entropy loss.
Activation Functions: Choose activation functions appropriate for your problem. For output layers in classification tasks, 'sigmoid' or softmax (if implemented) are common choices.

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Mustafa Bangash - mustafa22bangash@gmail.com
Project Link: https://github.com/MustafaBangash/SimpleCNN
