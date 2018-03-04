import numpy as np

class GradientDescent:
    learning_rate = 0.025

    def __init__(self, weights, biases, weight_gradients, bias_gradients):
        self.weights          = weights
        self.biases           = biases
        self.weight_gradients = weight_gradients
        self.bias_gradients   = bias_gradients

    def updated_parameters(self):
        updated_weights = self.weights - (self.learning_rate * self.weight_gradients)
        updated_biases  = self.biases  - (self.learning_rate * self.bias_gradients)

        return [updated_weights, updated_biases]

class Adam:
    learning_rate = 0.025
    momentum_rate = 0.9
    rms_prop_rate = 0.999
    epsilon       = 0.00000001

    def __init__(self, weights, biases, weight_gradients, bias_gradients, momentum_weight_average, momentum_bias_average, rms_prop_weight_average, rms_prop_bias_average):
        self.weights                 = weights
        self.biases                  = biases
        self.weight_gradients        = weight_gradients
        self.bias_gradients          = bias_gradients
        self.momentum_weight_average = momentum_weight_average
        self.momentum_bias_average   = momentum_bias_average
        self.rms_prop_weight_average = rms_prop_weight_average
        self.rms_prop_bias_average   = rms_prop_bias_average

    def updated_parameters(self):
        weight_denominator = np.array([
            np.sqrt(self.__updated_rms_prop_weight_average()[0]),
            np.sqrt(self.__updated_rms_prop_weight_average()[1]),
            np.sqrt(self.__updated_rms_prop_weight_average()[2])
        ]) + self.epsilon

        bias_denominator = np.array([
            np.sqrt(self.__updated_rms_prop_bias_average()[0]),
            np.sqrt(self.__updated_rms_prop_bias_average()[1]),
            np.sqrt(self.__updated_rms_prop_bias_average()[2])
        ]) + self.epsilon

        updated_weights = self.weights - (self.learning_rate * self.__updated_momentum_weight_average() / weight_denominator)
        updated_biases  = self.biases  - (self.learning_rate * self.__updated_momentum_bias_average()   / bias_denominator)

        return [
            updated_weights,
            updated_biases,
            self.__updated_momentum_weight_average(),
            self.__updated_momentum_bias_average(),
            self.__updated_rms_prop_weight_average(),
            self.__updated_rms_prop_bias_average()
        ]

    def __updated_momentum_weight_average(self):
        return (self.momentum_rate * self.momentum_weight_average) + ((1 - self.momentum_rate) * self.weight_gradients)

    def __updated_momentum_bias_average(self):
        return (self.momentum_rate * self.momentum_bias_average) + ((1 - self.momentum_rate) * self.bias_gradients)

    def __updated_rms_prop_weight_average(self):
        return (self.rms_prop_rate * self.rms_prop_weight_average) + ((1 - self.rms_prop_rate) * np.square(self.weight_gradients))

    def __updated_rms_prop_bias_average(self):
        return (self.rms_prop_rate * self.rms_prop_bias_average) + ((1 - self.rms_prop_rate) * np.square(self.bias_gradients))
