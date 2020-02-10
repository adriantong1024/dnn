import numpy as np


class NeuralLayer:
    def get_row(self):
        return self.weights.shape[0]

    def get_col(self):
        return self.weights.shape[1]

    def initialize_weights(self, identity):
        # randomly initialize the weights.
        for r in range(self.weights.shape[0]):
            for c in range(self.weights.shape[1]):
                if identity:
                    if r == c:
                        self.weights[r][c] = 1.0
                    else:
                        self.weights[r][c] = 0.0
                else:
                    self.weights[r][c] = np.random.normal()
        return

    def dump_weights(self):
        print(self.weights)

    def dump(self):
        print("This is the {} layer with dimension {}x1".format(
            self.layer_num, self.weights.shape[0]))
        self.dump_weights()

    def __init__(self, r, c, num, identity=False):
        # the number of the layer, start with zero.
        self.layer_num = num
        # weights of this layer.
        self.weights = np.array((0, 0), dtype=np.float)
        self.weights.resize((r, c))
        # initialize the weights.
        self.initialize_weights(identity)
        return
