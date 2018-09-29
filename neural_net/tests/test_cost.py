import unittest
import numpy as np

from lib.cost import Cost

class TestCost(unittest.TestCase):
    def setUp(self):
        # A 1 in a particular row indicates the example belongs to the class
        # with the index of that row. Eg given the labels below there are 3
        # classes: 0, 1, and 2, and a single example that belongs to class 1
        self.labels = np.array([
            [0], # row/class 0
            [1], # row/class 1 ... the 1 here indicates this is the example's class
            [0]  # row/class 2
        ])

        self.weights = np.array([
            [1], [1], [1],
            [1], [1], [1],
            [1], [1], [1]
        ])

        self.regularization_strength = 0.001

    def test_cost_is_very_high_for_incorrect_predictions_with_high_confidence(self):
        predictions = np.array([
            [0.90], # the network predicts a 90% chance that the example is of class 0
            [0.05],
            [0.05]
        ])
        cost = Cost(predictions,
                    self.labels,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

        self.assertTrue(cost > 2.5)

    def test_cost_is_very_low_for_correct_predictions_with_high_confidence(self):
        predictions = np.array([
            [0.05],
            [0.90], # the network predicts a 90% chance that the example is of class 1
            [0.05]
        ])
        cost = Cost(predictions,
                    self.labels,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

        self.assertTrue(cost < 0.2)

    def test_cost_is_moderate_for_correct_predictions_with_low_confidence(self):
        predictions = np.array([
            [0.25],
            [0.50], # the network predicts a 50% chance that the example is of class 1
            [0.25]
        ])
        cost = Cost(predictions,
                    self.labels,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

        self.assertTrue(cost < 0.7)
        self.assertTrue(cost > 0.5)

    def test_cost_is_high_for_incorrect_predictions_with_low_confidence(self):
        predictions = np.array([
            [0.50], # the network predicts a 50% chance that the example is of class 0
            [0.25],
            [0.25]
        ])
        cost = Cost(predictions,
                    self.labels,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

        self.assertTrue(cost < 1.5)
        self.assertTrue(cost > 1.0)

    def test_cost_increases_as_magnitude_of_weights_increases(self):
        predictions = np.array([
            [0.50], # the network predicts a 50% chance that the example is of class 0
            [0.25],
            [0.25]
        ])

        weights = np.array([
            [10], [10], [10],
            [10], [10], [10],
            [10], [10], [10]
        ])

        cost = Cost(predictions,
                    self.labels,
                    weights,
                    self.regularization_strength).cross_entropy_loss()

        self.assertTrue(cost < 2.0)
        self.assertTrue(cost > 1.5)

if __name__ == '__main__':
    unittest.main()
