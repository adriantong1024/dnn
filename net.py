import numpy as np
import neurallayer as nl


class NeuralNet:
    def __init__(self):
        # layers of the network.
        self.layers = []

    def add_single_layer(self, size):
        if len(self.layers) == 0:
            self.layers.append(nl.NeuralLayer(size, size, 0, True))
            return
        self.layers.append(
            nl.NeuralLayer(size, self.layers[-1].get_row(), len(self.layers)))
        return

    def feedforward(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].feedforward(self.layers[i].data)
                return
            self.layers[i].feedforward(self.layers[i - 1].data)

    def dump_layers(self):
        for layer in self.layers:
            layer.dump()

    def dump(self):
        self.dump_layers()
