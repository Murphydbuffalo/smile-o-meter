# import unittest
# import numpy as np
#
# from lib.optimizers.adam             import Adam
# from lib.optimizers.gradient_descent import GradientDescent
# from lib.optimize                    import Optimize
# from lib.data.raw                    import Raw
#
# learning_rate           = 0.01
# regularization_strength = 0.0001
# num_input_features      = 10
# num_hidden_layer_nodes  = 5
# num_classes             = 3
# num_examples            = 100
# batch_size              = 10
#
# # Can't pass a Numpy array to `hash`, so we first convert it to a tuple
# def hash_array(array):
#      return hash(tuple(array))
#
# class TestOptimize(unittest.TestCase):
#     def setUp(self):
#         self.weights = np.array([
#             np.random.randn(num_hidden_layer_nodes, num_input_features),
#             np.random.randn(num_classes, num_hidden_layer_nodes),
#         ])
#
#         self.biases = np.array([
#             np.zeros((num_hidden_layer_nodes, 1)),
#             np.zeros((num_classes, 1)),
#         ])
#         examples = np.random.randn(num_input_features, num_examples)
#         labels   = np.tile(np.array([
#             [1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
#             [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
#             [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
#         ]), 10)
#
#         gradient_descent = GradientDescent(learning_rate, self.weights, self.biases)
#         adam             = Adam(learning_rate, self.weights, self.biases)
#
#         self.gradient_descent_optimizer = Optimize(examples,
#                                                    labels,
#                                                    gradient_descent,
#                                                    regularization_strength,
#                                                    batch_size      = batch_size,
#                                                    logging_enabled = False)
#
#         self.adam_optimizer = Optimize(examples,
#                                        labels,
#                                        adam,
#                                        regularization_strength,
#                                        batch_size      = batch_size,
#                                        logging_enabled = False)
#
#     def test_returns_learned_parameters_with_gradient_descent(self):
#         result = self.gradient_descent_optimizer.run()
#
#         self.assertTrue(len(result['weights']) == len(self.weights))
#         self.assertTrue(result['weights'][0].shape == self.weights[0].shape)
#         self.assertTrue(result['weights'][1].shape == self.weights[1].shape)
#         self.assertFalse((result['weights'][1] == self.weights[1]).all())
#
#         self.assertTrue(len(result['biases']) == len(self.biases))
#         self.assertTrue(result['biases'][0].shape == self.biases[0].shape)
#         self.assertTrue(result['biases'][1].shape == self.biases[1].shape)
#         self.assertFalse((result['biases'][1] == self.biases[1]).all())
#
#     def test_generally_lowers_costs_with_gradient_descent(self):
#         result = self.gradient_descent_optimizer.run()
#         costs  = result['costs']
#
#         # Assert that cost decreases every 5th epoch
#         for epoch in range(1, int(len(costs) / 5)):
#             self.assertTrue(costs[epoch * 5] < costs[(epoch - 1) * 5])
#
#     def test_returns_learned_parameters_and_cost_with_adam(self):
#         result = self.adam_optimizer.run()
#
#         self.assertTrue(len(result['weights']) == len(self.weights))
#         self.assertTrue(result['weights'][0].shape == self.weights[0].shape)
#         self.assertTrue(result['weights'][1].shape == self.weights[1].shape)
#         self.assertFalse((result['weights'][1] == self.weights[1]).all())
#
#         self.assertTrue(len(result['biases']) == len(self.biases))
#         self.assertTrue(result['biases'][0].shape == self.biases[0].shape)
#         self.assertTrue(result['biases'][1].shape == self.biases[1].shape)
#         self.assertFalse((result['biases'][1] == self.biases[1]).all())
#
#     def test_generally_lowers_costs_with_adam(self):
#         result = self.adam_optimizer.run()
#         costs  = result['costs']
#
#         # Assert that cost decreases every 5th epoch
#         for epoch in range(1, int(len(costs) / 5)):
#             self.assertTrue(costs[epoch * 5] < costs[(epoch - 1) * 5])
#
#     def test_shuffle_in_unison(self):
#         raw_data  = Raw('./lib/data/sources/fer_subset.csv').load()
#         labels    = raw_data.training_labels
#         examples  = raw_data.training_examples
#         max_index = labels.shape[1]
#
#         label_to_example_mappings = {}
#
#         for i in range(max_index):
#             label_hash   = hash_array(labels[:,i])
#             example_hash = hash_array(examples[:,i])
#
#             label_to_example_mappings.setdefault(label_hash, [])
#             label_to_example_mappings[label_hash].append(example_hash)
#
#         self.adam_optimizer.shuffle_in_unison(labels, examples)
#
#         for i in range(max_index):
#             label_hash   = hash_array(labels[:,i])
#             example_hash = hash_array(examples[:,i])
#
#             self.assertTrue(example_hash in label_to_example_mappings[label_hash])
#
# if __name__ == '__main__':
#     unittest.main()
