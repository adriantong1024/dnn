from mlxtend.data import loadlocal_mnist
import numpy as np
import net as nn
import activation as act


def load_data(image_path, label_path):
    return loadlocal_mnist(image_path, label_path)


if __name__ == "__main__":
    # load the mnist dataset.
    X, Y = load_data('./data/train-images-idx3-ubyte',
                     './data/train-labels-idx1-ubyte')
    print("X dimension is {}x{}\n".format(X.shape[0], X.shape[1]))

    # build the neural network.
    dnn = nn.NeuralNet()
    dnn.add_single_layer(X.transpose().shape[0], act.Identity(), True,
                         X.transpose())
    dnn.add_single_layer(5, act.ReLU(), False)
    dnn.add_single_layer(10, act.ReLU(), False)
    dnn.add_single_layer(10, act.SoftMax(), True)
    dnn.feedforward()
    dnn.dump()

    # dnn.dump_layer(len(dnn.layers)-1)
    # layer=dnn.get_layer(len(dnn.layers)-1)
    # data=np.sum(layer.output, axis=0)
    # print (data, data.shape)
