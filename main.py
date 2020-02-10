from mlxtend.data import loadlocal_mnist
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


class NeuralNet:
    def __init__(self):
        # layers of the network.
        self.layers = []

    def add_single_layer(self, size):
        if len(self.layers) == 0:
            self.layers.append(NeuralLayer(size, size, 0, True))
            return
        self.layers.append(
            NeuralLayer(size, self.layers[-1].get_row(), len(self.layers)))
        return

    def dump_layers(self):
        for layer in self.layers:
            layer.dump()

    def dump(self):
        self.dump_layers()


def load_data(image_path, label_path):
    return loadlocal_mnist(image_path, label_path)


if __name__ == "__main__":
    # load the mnist dataset.
    X, Y = load_data('./data/train-images-idx3-ubyte',
                     './data/train-labels-idx1-ubyte')
    print("X dimension is {}x{}\n".format(X.shape[0], X.shape[1]))

    # build the neural network.
    dnn = NeuralNet()
    dnn.add_single_layer(5)
    dnn.add_single_layer(5)
    dnn.add_single_layer(1)
    dnn.dump()
