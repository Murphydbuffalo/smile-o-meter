import numpy as np

class ForwardProp:
    def __init__(self, weights, biases, X):
        self.weights               = weights
        self.biases                = biases
        self.X                     = X
        self.linear_activations    = []
        self.nonlinear_activations = [X]

    def run(self):
        A = self.X

        for i in range(len(self.weights)):
            Z = np.dot(self.weights[i], A)  + self.biases[i]
            A = self.__relu(Z)

            self.linear_activations.append(Z)
            self.nonlinear_activations.append(A)

        final_activation               = np.atleast_2d(np.apply_along_axis(self.__softmax, 0, Z))
        self.nonlinear_activations[-1] = final_activation

        return [final_activation, self.linear_activations, self.nonlinear_activations]

    # TODO: Try softplus to see if it performs noticeably better or worse.
    def __relu(self, Z):
        # NOTE: np.maximum is NOT the same as np.max, which find the maximum value
        # in the matrix (or in each row/column of the matrix).
        # np.maximum maps the given array element-wise, returning the max of each
        # element and the provided second arg (0 in this case)
        return np.maximum(Z, 0)

    def __softmax(self, v):
        exponentials = np.exp(v)
        return exponentials / np.sum(exponentials)
