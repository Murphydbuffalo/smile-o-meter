import unittest
import numpy as np

from lib.forward_prop import ForwardProp
from lib.predict      import Predict

class TestPredict(unittest.TestCase):
    def setUp(self):
        examples = np.random.randn(5, 2)

        # A 1 in a particular row indicates the example belongs to the class
        # with the index of that row. Eg given the labels below there are 3
        # classes: 0, 1, and 2, and two examples. The first example belongs to
        # class 1, and the second belongs to class 2.
        labels = np.array([
            [0, 0], # row/class 0
            [1, 0], # row/class 1, the first example (column) belongs to this class
            [0, 1]  # row/class 2, the second example (column) belongs to this class
        ])

        weights = np.array([np.random.randn(3, 5)])
        biases  = np.array([np.zeros((3, 1))])

        self.mock_network_output = np.array([
            [0.10, 0.80], # network predicts an 80% chance example 2 belongs to this class
            [0.80, 0.10], # network predicts an 80% chance example 1 belongs to this class
            [0.10, 0.10]
        ])

        self.predictor = Predict(examples, labels, weights, biases)
        self.predictor.run()

    def test_num_examples(self):
        self.predictor.network_output = self.mock_network_output
        self.assertEqual(self.predictor.num_examples, 2)

    def test_num_correct(self):
        self.predictor.network_output = self.mock_network_output
        self.assertEqual(self.predictor.num_correct(), 1)

    def test_percent_correct(self):
        self.predictor.network_output = self.mock_network_output
        self.assertEqual(self.predictor.percent_correct(), 0.5)

if __name__ == '__main__':
    unittest.main()
