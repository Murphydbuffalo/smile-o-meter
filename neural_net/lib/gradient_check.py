import numpy as np

from lib.forward_prop import ForwardProp
from lib.cost         import Cost

class GradientCheck:
    def __init__(self, weights, biases, examples, labels, regularization_strength):
        self.weights                 = weights
        self.biases                  = biases
        self.examples                = examples
        self.labels                  = labels
        self.regularization_strength = regularization_strength
        self.epsilon                 = 0.00001
        self.gradients               = []

        for i in range(len(weights)):
            self.gradients.append(np.zeros(weights[i].shape))

    def numeric_gradients(self):
        for layer in range(0, len(self.weights)):
            for row in range(self.num_rows(layer)):
                for column in range(self.num_columns(layer)):
                    original_weight = self.weights[layer][row][column]

                    self.gradients[layer][row][column] = self.numeric_gradient(original_weight,
                                                                               layer,
                                                                               row,
                                                                               column)
                    self.weights[layer][row][column] = original_weight

        return self.gradients

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
