import numpy as np
def relu(z):
    return np.where(z >= 0, z, 0)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def tanh(z):
    return np.tanh(z)
def leaky_relu(z):
    return np.where(z >= 0, z, 0.1 * z)
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
def relu_dif(z):
    return np.where(z >= 0, 1, 0)
def leaky_relu_dif(z):
    return np.where(z >= 0, 1, 0.1)
def sigmoid_def(z):
    return sigmoid(z) * (1 - sigmoid(z))
def tanh_def(z):
    return 1 - (tanh(z)) ** 2
def softmax_def(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
