from mlxtend.data import loadlocal_mnist
import numpy as np
import neuralnet as nn


def load_data(image_path, label_path):
    return loadlocal_mnist(image_path, label_path)


if __name__ == "__main__":
    # load the mnist dataset.
    X, Y = load_data('./data/train-images-idx3-ubyte',
                     './data/train-labels-idx1-ubyte')
    print("X dimension is {}x{}\n".format(X.shape[0], X.shape[1]))

    # build the neural network.
    dnn = nn.NeuralNet()
    dnn.add_single_layer(5)
    dnn.add_single_layer(5)
    dnn.add_single_layer(1)
    dnn.dump()
