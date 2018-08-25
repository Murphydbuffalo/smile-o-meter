import unittest
import numpy as np

from lib.forward_prop import ForwardProp

num_input_features     = 10
num_hidden_layer_nodes = 5
num_classes            = 3
num_examples           = 1000

class TestForwardProp(unittest.TestCase):
    def setUp(self):
        weights = np.array([
            np.random.randn(num_hidden_layer_nodes, num_input_features),
            np.random.randn(num_classes, num_hidden_layer_nodes),
        ])

        biases = np.array([
            np.zeros((num_hidden_layer_nodes, 1)),
            np.zeros((num_classes, 1)),
        ])

        examples     = np.random.randn(num_input_features, num_examples)
        forward_prop = ForwardProp(weights, biases, examples)

        self.linear_activations, self.nonlinear_activations = forward_prop.run()
        self.output = forward_prop.network_output

    def test_network_output_same_as_last_nonlinear_activation(self):
        self.assertTrue((self.output == self.nonlinear_activations[-1]).all())

    def test_network_output_shape(self):
        self.assertEqual(self.output.shape, (num_classes, num_examples))

    def test_network_output_columnwise_sum_equals_1(self):
        self.assertTrue((self.output.sum(axis=0) > 0.99).all())
        self.assertTrue((self.output.sum(axis=0) < 1.01).all())

if __name__ == '__main__':
    unittest.main()
