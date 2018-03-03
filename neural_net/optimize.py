import numpy as np

class GradientDescent:
    learning_rate = 0.015

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
    learning_rate = 0.0025
    momentum_rate = 0.9
    rms_prop_rate = 0.999

    def __init__(self, weights, biases, weight_gradients, bias_gradients, momentum_weight_average, momentum_bias_average, rms_prop_weight_average, rms_prop_bias_average):
        self.weights                 = weights
        self.biases                  = biases
        self.weight_gradients        = weight_gradients
        self.bias_gradients          = bias_gradients
        self.momentum_weight_average = (self.momentum_rate * momentum_weight_average) + ((1 - self.momentum_rate) * weight_gradients)
        self.momentum_bias_average   = (self.momentum_rate * momentum_bias_average)   + ((1 - self.momentum_rate) * bias_gradients)
        self.rms_prop_weight_average = (self.rms_prop_rate * rms_prop_weight_average) + ((1 - self.rms_prop_rate) * (weight_gradients * weight_gradients))
        self.rms_prop_bias_average   = (self.rms_prop_rate * rms_prop_bias_average)   + ((1 - self.rms_prop_rate) * (bias_gradients * bias_gradients))

    def updated_parameters(self):
        updated_weights = self.weights - (self.learning_rate * (self.momentum_weight_average / np.sqrt(self.rms_prop_weight_average)))
        updated_biases  = self.biases  - (self.learning_rate * (self.momentum_bias_average / np.sqrt(self.rms_prop_bias_average)))

        return [updated_weights, updated_biases]
