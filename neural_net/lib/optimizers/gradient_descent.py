import numpy as np

class GradientDescent:
    learning_rate = 0.001

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases  = biases

    def update_parameters(self, weight_gradients, bias_gradients):
        self.weights = self.weights - (self.learning_rate * weight_gradients)
        self.biases  = self.biases  - (self.learning_rate * bias_gradients)
