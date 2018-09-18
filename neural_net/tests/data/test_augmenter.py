import unittest
import numpy as np

from imgaug import augmenters

from lib.data.raw       import Raw
from lib.data.augmenter import Augmenter

class TestAugmenter(unittest.TestCase):
    def setUp(self):
        data                   = Raw('./lib/data/sources/fer_subset.csv').load()
        self.original_examples = data.training_examples
        self.original_labels   = data.training_labels
        self.augmenter         = Augmenter(self.original_examples, self.original_labels).augment()

    def test_additional_examples_are_generated(self):
        num_augmented = self.augmenter.augmented_examples.shape[1]
        num_original  = self.original_examples.shape[1]
        self.assertEqual(num_augmented, num_original * 6)

    def test_additional_examples_are_valid(self):
        augmented                  = self.augmenter.augmented_examples
        original                   = self.original_examples
        num_features, num_examples = original.shape

        # The original examples should be included
        self.assertTrue((original == augmented[:, 0:num_examples]).all())

        # As well as transformations of the original examples
        formatted_images     = self.augmenter.format_for_augmentation(original)
        flipper              = augmenters.Sequential([augmenters.Fliplr(1)])
        flipped              = flipper.augment_images(formatted_images)
        first_transformation = flipped.reshape((num_examples, num_features)).T

        self.assertTrue(
            (first_transformation == augmented[:, num_examples:num_examples * 2]).all()
        )

    def test_additional_labels_are_generated(self):
        num_augmented = self.augmenter.augmented_labels.shape[1]
        num_original  = self.original_labels.shape[1]
        self.assertEqual(num_augmented, num_original * 6)

    # `Augmenter` should copy the labels once for each transformation of the
    # examples it generates.
    def test_additional_labels_are_valid(self):
        augmented  = self.augmenter.augmented_labels
        original   = self.original_labels
        num_labels = original.shape[1]

        self.assertTrue((original == augmented[:, 0:num_labels]).all())
        self.assertTrue((original == augmented[:, num_labels:num_labels * 2]).all())

if __name__ == '__main__':
    unittest.main()
