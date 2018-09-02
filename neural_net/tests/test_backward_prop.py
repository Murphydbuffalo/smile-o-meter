import unittest
import numpy as np
import pdb

from lib.backward_prop  import BackwardProp
from lib.forward_prop   import ForwardProp
from lib.gradient_check import GradientCheck

num_input_features     = 10
num_hidden_layer_nodes = 5
num_classes            = 3
num_examples           = 1000

class TestBackwardProp(unittest.TestCase):
    def setUp(self):
        weights = np.array([
            np.random.randn(num_hidden_layer_nodes, num_input_features),
            np.random.randn(num_classes, num_hidden_layer_nodes),
        ])

        biases = np.array([
            np.zeros((num_hidden_layer_nodes, 1)),
            np.zeros((num_classes, 1)),
        ])

        random_values = np.random.randn(num_classes, num_examples)
        labels = (random_values == random_values.max(axis=0)) * 1

        examples = np.random.randn(num_input_features, num_examples)

        forward_prop = ForwardProp(weights, biases, examples)
        linear_activations, nonlinear_activations = forward_prop.run()

        regularization_strength = 0.001

        self.backward_prop = BackwardProp(weights,
                                          linear_activations,
                                          nonlinear_activations,
                                          labels,
                                          regularization_strength)

        gradient_check = GradientCheck(weights,
                                       biases,
                                       examples,
                                       labels,
                                       regularization_strength)

        self.numeric_gradients = gradient_check.numeric_gradients()

    # The job of `BackwardProp` is to efficiently calculate gradients (
    # derivatives for a vector/array). A neural network can easily have many
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
    def test_analytics_gradients_are_the_same_as_numeric_gradients(self):
        weight_gradients, bias_gradients = self.backward_prop.run()

        self.assertTrue(
            np.allclose(
                weight_gradients[-1],
                self.numeric_gradients[-1],
                atol = 0.005
            )
        )

        self.assertTrue(
            np.allclose(
                weight_gradients[-2],
                self.numeric_gradients[-2],
                atol = 0.005
            )
        )

if __name__ == '__main__':
    unittest.main()
