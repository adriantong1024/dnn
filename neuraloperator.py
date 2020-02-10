import numpy as np


class Sigmoid:
    def __init__(self):
        return

    def feedforward(self, data):
        # 1/(1+e^-z)
        return np.power(1 + np.exp(data), -1)

    def backward(self, data):
        return data
