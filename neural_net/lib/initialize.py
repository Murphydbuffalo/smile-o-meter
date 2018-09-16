import numpy as np

class Initialize:
    xavier_numerator = 1
    he_numerator     = 2

    def __init__(self, network_shape, algorithm = 'xavier'):
        self.network_shape = network_shape
        self.algorithm     = algorithm

    def weights_and_biases(self):
        weights = []
        biases  = []

        for i in range(0, len(self.network_shape) - 1):
            num_inputs  = self.network_shape[i]
            num_outputs = self.network_shape[i + 1]

            weights.append(self.initialize_weights(num_inputs, num_outputs))
            biases.append(self.initialize_biases(num_outputs))

        return [np.array(weights), np.array(biases)]

    def initialize_weights(self, num_inputs, num_outputs):
        if self.algorithm == 'xavier':
            numerator = self.xavier_numerator
        else:
            numerator = self.he_numerator

        return np.random.randn(num_outputs, num_inputs) * np.sqrt(numerator / num_inputs)

    def initialize_biases(self, num_outputs):
       return np.zeros((num_outputs, 1))
