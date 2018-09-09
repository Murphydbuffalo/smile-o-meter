import unittest
import numpy as np

from lib.optimizers.adam import Adam

num_input_features     = 10
num_hidden_layer_nodes = 5
num_classes            = 3

class TestAdam(unittest.TestCase):
    # Adam is a "momentum" based optimizer. It's similar to GradientDescent in
    # that it updates parameters by subtracting the gradients (derivatives) of
    # those parameters multiplied by some small coefficient (the learning rate).
    # Let's call that product the "update amount".
    #
    # Adam differs from GradientDescent in that its update amount also includes
    # a weighted average of previous update amounts. So, as you iteratively call
    # `Adam#update_parameters(weight_gradients, bias_gradients)`, the update
    # amount is influenced more and more by its previous values, and less by the
    # gradients passed in to that iteration. In this way Adam "builds momentum"
    # by updating parameters in a way that over time becomes less responsive to
    # the arguments passed in, and more similar to previous updates. This method
    # is far more effective at quickly finding optimal paramters than non-momentum
    # based algorithms like GradientDescent.
    def test_parameters_decrease_at_a_decreasing_rate(self):
        weights = np.array([
            np.ones((num_hidden_layer_nodes, num_input_features)) * 0.01,
            np.ones((num_classes, num_hidden_layer_nodes))        * 0.01
        ])

        biases = np.array([
            np.zeros((num_hidden_layer_nodes, 1)),
            np.zeros((num_classes, 1))
        ])

        weight_gradients = np.copy(weights)

        bias_gradients = np.array([
            np.random.randn(num_hidden_layer_nodes, 1),
            np.random.randn(num_classes, 1)
        ])

        adam = Adam(weights, biases)

        update_amounts = []
        for i in range(0, 10):
            weights_before_update = adam.weights
            adam.update_parameters(weight_gradients, bias_gradients)
            update_amounts.append(abs(weights_before_update[0] - adam.weights[0]).mean())

        update_amount_deltas = []
        for i in range(0, len(update_amounts) - 1):
            self.assertTrue(update_amounts[i] < update_amounts[i + 1])
            update_amount_deltas.append(abs(update_amounts[i] - update_amounts[i + 1]))

        for i in range(0, len(update_amount_deltas) - 1):
            self.assertTrue(update_amount_deltas[i] > update_amount_deltas[i + 1])

if __name__ == '__main__':
    unittest.main()
