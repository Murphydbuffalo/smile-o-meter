import unittest
import numpy as np

from lib.optimizers.gradient_descent import GradientDescent

learning_rate          = 0.001
num_input_features     = 10
num_hidden_layer_nodes = 5
num_classes            = 3

class TestGradientDescent(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([
            np.random.randn(num_hidden_layer_nodes, num_input_features),
            np.random.randn(num_classes, num_hidden_layer_nodes),
        ])

        self.biases = np.array([
            np.zeros((num_hidden_layer_nodes, 1)),
            np.zeros((num_classes, 1)),
        ])

        self.gradient_descent = GradientDescent(learning_rate, self.weights, self.biases)

    def test_update_parameters(self):
        weight_gradients = np.array([
            np.ones((num_hidden_layer_nodes, num_input_features)),
            np.ones((num_classes, num_hidden_layer_nodes)),
        ])

        bias_gradients = np.array([
            np.random.randn(num_hidden_layer_nodes, 1),
            np.random.randn(num_classes, 1),
        ])

        self.gradient_descent.update_parameters(weight_gradients, bias_gradients)

        expected_updated_weights = self.weights - learning_rate

        self.assertTrue(
            np.allclose(self.gradient_descent.weights[0], expected_updated_weights[0])
        )

        self.assertTrue(
            np.allclose(self.gradient_descent.weights[1], expected_updated_weights[1])
        )

if __name__ == '__main__':
    unittest.main()
