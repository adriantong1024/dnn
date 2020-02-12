import numpy as np


class Identity:
    def __init__(self):
        return

    def compute(self, data):
        return data

    def feedforward(self, data):
        return self.compute(data)


class Sigmoid:
    def __init__(self):
        return

    def compute(self, data):
        # 1/(1+e^-z)
        return np.power(1 + np.exp(data), -1)

    def feedforward(self, data):
        return self.compute(data)

    def backward(self, data):
        return np.multiply(self.compute(data), (1 - self.compute(data)))


class ReLU:
    def __init__(self):
        return

    def feedforward(self, data):
        # relu(x) is max(0, x)
        zeros = np.zeros(data.shape, dtype=np.float)
        return np.maximum(data, zeros)

    def backward(self, data):
        return
