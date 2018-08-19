import numpy as np

class Adam:
    learning_rate = 0.0005
    momentum_rate = 0.9
    rms_prop_rate = 0.999
    epsilon       = 0.00000001

    def __init__(self, weights, biases):
        self.weights                 = weights
        self.biases                  = biases
        self.momentum_weight_average = np.zeros(weights.shape)
        self.momentum_bias_average   = np.zeros(biases.shape)
        self.rms_prop_weight_average = np.zeros(weights.shape)
        self.rms_prop_bias_average   = np.zeros(biases.shape)

    def update_parameters(self, weight_gradients, bias_gradients):
        self.momentum_weight_average = self.__updated_momentum_weight_average(weight_gradients)
        self.momentum_bias_average   = self.__updated_momentum_bias_average(bias_gradients)
        self.rms_prop_weight_average = self.__updated_rms_prop_weight_average(weight_gradients)
        self.rms_prop_bias_average   = self.__updated_rms_prop_bias_average(bias_gradients)

        self.weights = self.weights - (self.learning_rate * self.momentum_weight_average / self.__weight_denominator())
        self.biases  = self.biases  - (self.learning_rate * self.momentum_bias_average   / self.__bias_denominator())

    def __updated_momentum_weight_average(self, weight_gradients):
        return (self.momentum_rate * self.momentum_weight_average) + ((1 - self.momentum_rate) * weight_gradients)

    def __updated_momentum_bias_average(self, bias_gradients):
        return (self.momentum_rate * self.momentum_bias_average) + ((1 - self.momentum_rate) * bias_gradients)

    def __updated_rms_prop_weight_average(self, weight_gradients):
        return (self.rms_prop_rate * self.rms_prop_weight_average) + ((1 - self.rms_prop_rate) * np.square(weight_gradients))

    def __updated_rms_prop_bias_average(self, bias_gradients):
        return (self.rms_prop_rate * self.rms_prop_bias_average) + ((1 - self.rms_prop_rate) * np.square(bias_gradients))

    def __weight_denominator(self):
        # Why can't we call np.sqrt on an n-dimensional array?
        return np.array([
            np.sqrt(self.rms_prop_weight_average[0]),
            np.sqrt(self.rms_prop_weight_average[1]),
            np.sqrt(self.rms_prop_weight_average[2])
        ]) + self.epsilon

    def __bias_denominator(self):
        return np.array([
            np.sqrt(self.rms_prop_bias_average[0]),
            np.sqrt(self.rms_prop_bias_average[1]),
            np.sqrt(self.rms_prop_bias_average[2])
        ]) + self.epsilon
