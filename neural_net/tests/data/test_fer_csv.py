import unittest

from lib.data.sources.fer_csv import FER_CSV

num_pixels  = 48 * 48
num_classes = 7

class TestFER_CSV(unittest.TestCase):
    def setUp(self):
        self.data = FER_CSV('./lib/data/sources/fer_subset.csv').load_data()

    def test_examples_shape(self):
        self.assertEqual(self.data.training_examples.shape[0],   num_pixels)
        self.assertEqual(self.data.validation_examples.shape[0], num_pixels)
        self.assertEqual(self.data.test_examples.shape[0],       num_pixels)

    def test_labels_shape(self):
        self.assertEqual(self.data.training_labels.shape[0],   num_classes)
        self.assertEqual(self.data.validation_labels.shape[0], num_classes)
        self.assertEqual(self.data.test_labels.shape[0],       num_classes)

    def test_same_number_of_labels_and_examples(self):
        self.assertEqual(
            self.data.training_examples.shape[1],
            self.data.training_labels.shape[1]
        )

        self.assertEqual(
            self.data.validation_examples.shape[1],
            self.data.validation_labels.shape[1]
        )

        self.assertEqual(
            self.data.test_examples.shape[1],
            self.data.test_labels.shape[1]
        )

    def test_labels_values(self):
        self.assertEqual(
            self.data.training_labels.shape[1],
            self.data.training_labels.sum()
        )

        self.assertEqual(
            self.data.validation_labels.shape[1],
            self.data.validation_labels.sum()
        )

        self.assertEqual(
            self.data.test_labels.shape[1],
            self.data.test_labels.sum()
        )

if __name__ == '__main__':
    unittest.main()
