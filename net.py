import numpy as np
import layer as nl


class NeuralNet:
    def __init__(self):
        # layers of the network.
        self.layers = []

    def add_single_layer(self,
                         size,
                         activation,
                         identity,
                         output=np.array((0, 0), dtype=np.float)):
        if len(self.layers) == 0:
            self.layers.append(
                nl.NeuralLayer(size, size, 0, activation, identity, output))
            return
        self.layers.append(
            nl.NeuralLayer(size, self.layers[-1].get_row(), len(self.layers),
                           activation, identity, output))
        return

    def feedforward(self):
        for i in range(len(self.layers)):
            if i == 0:
                continue
            self.layers[i].feedforward(self.layers[i - 1].output)

    def dump_layers(self):
        for layer in self.layers:
            layer.dump()

    def dump(self):
        self.dump_layers()
