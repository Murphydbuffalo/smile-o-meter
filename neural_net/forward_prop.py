import numpy as np

class ForwardProp:
    def __init__(self, weights, biases, X):
        self.weights = weights
        self.biases  = biases
        self.X       = X

    def run(self):
        Z1 = np.dot(self.weights[0], self.X)  + self.biases[0]
        A1 = self.__relu(Z1)

        Z2 = np.dot(self.weights[1], A1) + self.biases[1]
        A2 = self.__relu(Z2)

        Z3 = np.dot(self.weights[2], A2) + self.biases[2]
        A3 = self.__relu(Z3)

        Z4 = np.dot(self.weights[3], A3) + self.biases[3]
        A4 = self.__sigmoid(Z4)

        return A4

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
