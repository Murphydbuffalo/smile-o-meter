import numpy as np

from lib.cost         import Cost
from lib.forward_prop import ForwardProp

# The job of `BackwardProp` is to *efficiently* calculate gradients
# (derivatives for a vector/array). A neural network can easily have many
# thousands of weights, and needs to calculate the derivative for all of
# in order to "learn" (update the weights in such a way that the network
# gets better at its assigned task). These weight updates will likely need
# to happen thousands of times before the network becomes suitably effective.
#
# Therefore it becomes prohibitively expensive to calculate the *numeric*
# gradients for each individual weight on every iteration. "Numeric" meaning
# calcuating the approximate derivative based on the two-sided limit formula
# `(f(x + e) - f(x - e)) / 2e`.
#
# Instead, we can calculate the "analytic" derivatives, which means
# algebraically calculating the derivatives for all weights in a layer at
# once, via the chain rule.
class GradientCheck:
    def __init__(self, weight_gradients, weights, biases, examples, labels, regularization_strength):
        self.weight_gradients        = weight_gradients
        self.weights                 = weights
        self.biases                  = biases
        self.examples                = examples
        self.labels                  = labels
        self.regularization_strength = regularization_strength
        self.epsilon                 = 0.0000001

    def run(self, layer):
        gradients = self.numeric_gradients(layer)
        return np.allclose(self.weight_gradients[layer], gradients, atol = 0.003)

    def numeric_gradients(self, layer):
        gradients = np.zeros(self.weights[layer].shape)

        for row in range(self.num_rows(layer)):
            for column in range(self.num_columns(layer)):
                original_weight = self.weights[layer][row][column]

                gradients[row][column] = self.numeric_gradient(original_weight,
                                                               layer,
                                                               row,
                                                               column)

                self.weights[layer][row][column] = original_weight

        return gradients

    def numeric_gradient(self, weight, layer, row, column):
        self.weights[layer][row][column]       = weight + self.epsilon
        network_output_with_weight_adjusted_up = self.network_output()

        self.weights[layer][row][column]         = weight - self.epsilon
        network_output_with_weight_adjusted_down = self.network_output()

        cost_with_weight_adjusted_up   = self.cost(network_output_with_weight_adjusted_up)
        cost_with_weight_adjusted_down = self.cost(network_output_with_weight_adjusted_down)

        cost_difference = (cost_with_weight_adjusted_up - cost_with_weight_adjusted_down)

        return cost_difference / (2 * self.epsilon)

    def cost(self, network_output):
        return Cost(network_output,
                    self.labels,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

    def network_output(self):
        forward_prop = ForwardProp(self.weights, self.biases, self.examples)
        forward_prop.run()

        return forward_prop.network_output

    def num_rows(self, layer):
        return self.weights[layer].shape[0]

    def num_columns(self, layer):
        return self.weights[layer].shape[1]
