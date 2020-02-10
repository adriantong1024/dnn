import numpy as np


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
