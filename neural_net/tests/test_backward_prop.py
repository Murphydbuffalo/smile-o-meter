import unittest
import numpy as np

from lib.forward_prop                import ForwardProp
from lib.cost                        import Cost
from lib.backward_prop               import BackwardProp
from lib.optimizers.gradient_descent import GradientDescent

num_input_features     = 10
num_hidden_layer_nodes = 5
num_classes            = 3
num_examples           = 1000

class TestBackwardProp(unittest.TestCase):
    def setUp(self):
        random_values                = np.random.randn(num_classes, num_examples)
        self.labels                  = (random_values == random_values.max(axis=0)) * 1
        self.examples                = np.random.randn(num_input_features, num_examples)
        self.regularization_strength = 0.001

    def test_analytic_gradients_are_close_to_numeric_gradients(self):
        weights = np.array([
            np.random.randn(num_hidden_layer_nodes, num_input_features),
            np.random.randn(num_classes, num_hidden_layer_nodes),
        ])

        biases = np.array([
            np.zeros((num_hidden_layer_nodes, 1)),
            np.zeros((num_classes, 1)),
        ])

        for i in range(0, 3):
            print('Testing backprop, iteration', i + 1)

            forward_prop = ForwardProp(weights, biases, self.examples)
            linear_activations, nonlinear_activations = forward_prop.run()

            backward_prop = BackwardProp(weights,
                                         linear_activations,
                                         nonlinear_activations,
                                         self.labels,
                                         self.regularization_strength)

            gradient_check = GradientCheck(weights,
                                           biases,
                                           self.examples,
                                           self.labels,
                                           self.regularization_strength)

            weight_gradients, bias_gradients = backward_prop.run()
            numeric_gradients                = gradient_check.numeric_gradients()

            self.assertTrue(
                np.allclose(
                    weight_gradients[-1],
                    numeric_gradients[-1],
                    atol = 0.005
                )
            )

            self.assertTrue(
                np.allclose(
                    weight_gradients[-2],
                    numeric_gradients[-2],
                    atol = 0.005
                )
            )

            optimizer = GradientDescent(weights, biases)
            optimizer.update_parameters(weight_gradients, bias_gradients)
            weights = optimizer.weights
            biases  = optimizer.biases


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

if __name__ == '__main__':
    unittest.main()
