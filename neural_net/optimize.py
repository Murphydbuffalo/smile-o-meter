class GradientDescent:
    learning_rate = 0.0000000025

    def __init__(self, weights, biases, weight_gradients, bias_gradients):
        self.weights          = weights
        self.biases           = biases
        self.weight_gradients = weight_gradients
        self.bias_gradients   = bias_gradients

    def updated_parameters(self):
        updated_weights = self.weights - (self.learning_rate * self.weight_gradients)
        updated_biases  = self.biases  - (self.learning_rate * self.bias_gradients)

        return [updated_weights, updated_biases]
