import numpy as np

class ForwardProp:
    def __init__(self, weights, biases, X):
        self.weights = weights
        self.biases  = biases
        self.X       = X

    def run(self):
        A = self.X

        for i in range(0, len(self.weights)):
            Z = np.dot(self.weights[i], A)  + self.biases[i]
            A = self.__relu(Z)

        return = self.__sigmoid(Z)

    def __relu(self, Z):
        # NOTE: np.maximum is NOT the same as np.max, which find the maximum value
        # in the matrix (or in each row/column of the matrix).
        # np.maximum maps the given array element-wise, returning the max of each
        # element and the provided second arg (0 in this case)
        return np.maximum(Z, 0)

    # Really this should be softmax, because you are predicting between 3 classes
    # Unhappy, neutral, and happy
    def __sigmoid(self, Z):
        return 1 / (1 + np.e ** -Z)