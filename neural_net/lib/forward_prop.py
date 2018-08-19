import numpy as np

class ForwardProp:
    def __init__(self, weights, biases, examples):
        self.weights               = weights
        self.biases                = biases
        self.examples              = examples
        self.linear_activations    = []
        self.nonlinear_activations = [examples]

    def run(self):
        for i in range(len(self.weights)):
            linear_activation    = np.dot(self.weights[i], self.nonlinear_activations[i])  + self.biases[i]
            nonlinear_activation = self.__relu(linear_activation)

            self.linear_activations.append(linear_activation)
            self.nonlinear_activations.append(nonlinear_activation)

        self.network_output            = self.__softmax_activation(linear_activation)
        self.nonlinear_activations[-1] = self.network_output

        return [self.linear_activations, self.nonlinear_activations]

    def __relu(self, linear_activation):
        # NOTE: np.maximum is NOT the same as np.max
        # np.max finds the maximum value in the matrix (or in each row/column
        # of the matrix)
        #
        # np.maximum maps the given array element-wise, returning the max of
        # each element and the provided second arg (0 in this case)
        return np.maximum(linear_activation, 0)

    def __softmax(self, vector):
        exponentials = np.exp(vector - np.max(vector, axis = 0))
        return exponentials / np.sum(exponentials)

    def __softmax_activation(self, linear_activation):
        return np.atleast_2d(np.apply_along_axis(self.__softmax, 0, linear_activation))
