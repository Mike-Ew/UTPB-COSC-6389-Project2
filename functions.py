import numpy as np

# ---------------------------------------
# Activation Functions and Derivatives
# ---------------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    # a is the output from sigmoid(z)
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(a):
    # derivative of tanh given output a
    return 1 - a**2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(a):
    # derivative of ReLU given output a
    return np.where(a > 0, 1, 0)


# ---------------------------------------
# Cost Functions and Derivatives
# ---------------------------------------


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def mse_derivative(predictions, targets):
    return predictions - targets


def cross_entropy(predictions, targets):
    # For binary classification:
    # predictions and targets expected to be of shape (N,) or (N,1)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-9
    return -np.mean(
        targets * np.log(predictions + epsilon)
        + (1 - targets) * np.log(1 - predictions + epsilon)
    )


def cross_entropy_derivative(predictions, targets):
    epsilon = 1e-9
    return (predictions - targets) / ((predictions * (1 - predictions)) + epsilon)
