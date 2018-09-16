import numpy as np
from lib.data.augmenter import Augmenter

class Formatter:
    def __init__(self, data):
        self.training_examples = data.training_examples
        self.training_labels   = data.training_labels
        self.test_examples     = data.test_examples
        self.test_labels       = data.test_labels
        self.num_features      = self.training_examples.shape[0]
        self.num_classes       = self.training_labels.shape[0]

    def run(self):
        self.remove_bad_data()
        self.augment()
        self.normalize()

        return self

    def remove_bad_data(self):
        training_examples, training_labels = self.remove_zero_standard_deviation_data(self.training_examples, self.training_labels)
        self.training_examples = training_examples
        self.training_labels   = training_labels

        validation_examples, validation_labels = self.remove_zero_standard_deviation_data(self.validation_examples, self.validation_labels)
        self.validation_examples = validation_examples
        self.validation_labels   = validation_labels

        test_examples, test_labels = self.remove_zero_standard_deviation_data(self.test_examples, self.test_labels)
        self.test_examples = test_examples
        self.test_labels   = test_labels

    def augment(self):
        augmenter              = Augmenter(self.training_examples, self.training_labels).augment()
        self.training_examples = augmenter.augmented_examples
        self.training_labels   = augmenter.augmented_labels

    def normalize(self):
        self.training_set_means                         = np.array([np.mean(self.training_examples, 1)]).T
        zero_mean_training_data                         = self.training_examples - self.training_set_means
        adjusted_mean_test_data                         = self.test_examples  - self.training_set_means
        self.zero_mean_training_set_standard_deviations = np.array([np.std(zero_mean_training_data, 1)]).T

        # Transform training data so it has mean 0 and variance 1, apply identical transformation to test data
        self.training_examples = (zero_mean_training_data) / self.zero_mean_training_set_standard_deviations
        self.test_examples     = (adjusted_mean_test_data) / self.zero_mean_training_set_standard_deviations

    def remove_zero_standard_deviation_data(self, matrix, labels):
        standard_deviations                   = np.std(matrix, 0)
        columns_with_zero_standard_devivation = np.where(standard_deviations == 0)[0]

        return [
            np.delete(matrix, columns_with_zero_standard_devivation, axis = 1),
            np.delete(labels, columns_with_zero_standard_devivation, axis = 1)
        ]
