import unittest
import numpy as np

from lib.forward_prop                import ForwardProp
from lib.cost                        import Cost
from lib.backward_prop               import BackwardProp
from lib.gradient_check              import GradientCheck
from lib.optimizers.gradient_descent import GradientDescent

learning_rate          = 0.01
num_input_features     = 28
num_hidden_layer_nodes = 14
num_classes            = 7
num_examples           = 100

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

            weight_gradients, bias_gradients = backward_prop.run()

            gradient_check = GradientCheck(weight_gradients,
                                           weights,
                                           biases,
                                           self.examples,
                                           self.labels,
                                           self.regularization_strength)

            for layer in range(len(weights)):
                self.assertTrue(gradient_check.run(layer))

            optimizer = GradientDescent(learning_rate, weights, biases)
            optimizer.update_parameters(weight_gradients, bias_gradients)
            weights = optimizer.weights
            biases  = optimizer.biases

if __name__ == '__main__':
    unittest.main()
