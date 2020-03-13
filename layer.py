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

    def feedforward(self, prev_output):
        # compute the this layer data.
        self.output = self.weights.dot(prev_output)
        # apply the nonlinear operator to the data of this layer.
        self.output = self.operator.feedforward(self.output)

    def feedbackward(self, prev_output):
        # feed through the activation function.
        self.output=self.operator.feedbackward(self.output)
        # calculate grad_weight. i.e. how each weight change to decrease L.
        

    def initialize_output(self, output):
        self.output = output

    def __init__(self, r, c, num, activation, identity, output):
        # the number of the layer, start with zero.
        self.layer_num = num
        # weights of this layer.
        self.weights = np.array((0, 0), dtype=np.float)
        self.weights.resize((r, c))

        self.grad_weights = np.array((0, 0), dtype=np.float)
        self.grad_weights.resize((r, c))
        # initialize the weights.
        self.initialize_weights(identity)

        # initialze operator. sigmoid for now.
        self.operator = activation

        # initialize data.
        self.output = output

    def dump_weights(self):
        print("This layer weight with dimension {}x{} is \n".format(
            self.weights.shape[0], self.weights.shape[1]), self.weights)

    def dump_output(self):
        print("This layer output with dimension {}x{} is\n".format(
            self.output.shape[0], self.output.shape[1]), self.output)

    def dump(self):
        print("\nThis is the {} layer".format(self.layer_num))
        self.dump_weights()
        self.dump_output()


