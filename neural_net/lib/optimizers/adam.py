import numpy as np

class Adam:
    momentum_rate = 0.9
    rms_prop_rate = 0.999
    epsilon       = 0.00000001

    def __init__(self, learning_rate, weights, biases):
        self.learning_rate           = learning_rate
        self.weights                 = weights
        self.biases                  = biases
        self.momentum_weight_average = np.zeros(weights.shape)
        self.momentum_bias_average   = np.zeros(biases.shape)
        self.rms_prop_weight_average = np.zeros(weights.shape)
        self.rms_prop_bias_average   = np.zeros(biases.shape)

    def update_parameters(self, weight_gradients, bias_gradients):
        self.momentum_weight_average = self.updated_momentum_weight_average(weight_gradients)
        self.momentum_bias_average   = self.updated_momentum_bias_average(bias_gradients)
        self.rms_prop_weight_average = self.updated_rms_prop_weight_average(weight_gradients)
        self.rms_prop_bias_average   = self.updated_rms_prop_bias_average(bias_gradients)

        self.weights = self.weights - (self.learning_rate * self.momentum_weight_average / self.weight_denominator())
        self.biases  = self.biases  - (self.learning_rate * self.momentum_bias_average   / self.bias_denominator())

    def updated_momentum_weight_average(self, weight_gradients):
        return (self.momentum_rate * self.momentum_weight_average) + ((1 - self.momentum_rate) * weight_gradients)

    def updated_momentum_bias_average(self, bias_gradients):
        return (self.momentum_rate * self.momentum_bias_average) + ((1 - self.momentum_rate) * bias_gradients)

    def updated_rms_prop_weight_average(self, weight_gradients):
        return (self.rms_prop_rate * self.rms_prop_weight_average) + ((1 - self.rms_prop_rate) * np.square(weight_gradients))

    def updated_rms_prop_bias_average(self, bias_gradients):
        return (self.rms_prop_rate * self.rms_prop_bias_average) + ((1 - self.rms_prop_rate) * np.square(bias_gradients))

    def weight_denominator(self):
        square_roots = self.square_roots(self.rms_prop_weight_average)
        return np.array(square_roots) + self.epsilon

    def bias_denominator(self):
        square_roots = self.square_roots(self.rms_prop_bias_average)
        return np.array(square_roots) + self.epsilon

    def square_roots(self, n_dimensional_array):
        return list(np.sqrt(layer) for layer in n_dimensional_array)
