import unittest
import numpy as np

from lib.data.normalizer import Normalizer

class TestFormatter(unittest.TestCase):
    def setUp(self):
        self.original_data = np.random.randn(10, 10) * 100
        self.normalizer    = Normalizer(self.original_data,
                                        self.original_data,
                                        self.original_data).normalize(save_statistics = False)

    def test_data_is_normalized_to_have_zero_mean_and_one_standard_deviation(self):
        self.assertFalse(np.isclose(self.original_data.mean(), 0, atol = 0.01))
        self.assertFalse(np.isclose(self.original_data.std(),  1, atol = 0.01))

        self.assertTrue(np.isclose(self.normalizer.normalized_training_examples.mean(), 0, atol = 0.01))
        self.assertTrue(np.isclose(self.normalizer.normalized_training_examples.std(),  1, atol = 0.01))

        self.assertTrue(np.isclose(self.normalizer.normalized_validation_examples.mean(), 0, atol = 0.01))
        self.assertTrue(np.isclose(self.normalizer.normalized_validation_examples.std(),  1, atol = 0.01))

        self.assertTrue(np.isclose(self.normalizer.normalized_test_examples.mean(), 0, atol = 0.01))
        self.assertTrue(np.isclose(self.normalizer.normalized_test_examples.std(),  1, atol = 0.01))

if __name__ == '__main__':
    unittest.main()
