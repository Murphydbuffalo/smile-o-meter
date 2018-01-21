import numpy as np
from forward_prop  import ForwardProp
from cost          import Cost

class GradientCheck:
    def __init__(self, weights, biases, weight_gradients, X, Y):
        self.weights          = weights
        self.biases           = biases
        self.weight_gradients = weight_gradients
        self.X                = X
        self.Y                = Y
        self.epsilon          = 0.00001

    def close_enough(self):
        acceptable_delta   = self.epsilon
        numeric_gradients  = self.__numeric_gradients()
        analytic_gradients = self.weight_gradients

        return np.abs(numeric_gradients - analytic_gradients).max() <= acceptable_delta

    def __numeric_gradients(self):
        weights           = np.copy(self.weights)
        numeric_gradients = np.copy(self.weight_gradients)

        for layer in range(len(weights)):
            for column in range(weights[layer].shape[0]):
                for row in range(weights[layer].shape[1]):
                    original_weight = weights[layer][column][row]

                    weights[layer][column][row] = original_weight + self.epsilon
                    Zplus, Aplus                = ForwardProp(weights, self.biases, self.X).run()


                    weights[layer][column][row] = original_weight - self.epsilon
                    Zminus, Aminus              = ForwardProp(weights, self.biases, self.X).run()

                    cost_plus  = Cost(Aplus[-1],  self.Y).cross_entropy_loss()
                    cost_minus = Cost(Aminus[-1], self.Y).cross_entropy_loss()

                    numeric_gradients[layer][column][row] = (cost_plus - cost_minus) / (2 * self.epsilon)

        return numeric_gradients
