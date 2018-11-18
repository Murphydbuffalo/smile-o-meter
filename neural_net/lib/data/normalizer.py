import numpy as np

class Normalizer:
    def __init__(self, training_examples, validation_examples, test_examples):
        self.training_examples   = training_examples
        self.validation_examples = validation_examples
        self.test_examples       = test_examples

    def normalize(self, save_statistics = True):
        training_set_means                         = self.training_examples.mean(axis = 1, keepdims = True)
        zero_mean_training_data                    = self.training_examples - training_set_means
        zero_mean_training_set_standard_deviations = zero_mean_training_data.std(axis = 1, keepdims = True)

        if save_statistics:
            print("Saving training set statistics for normalization...")
            np.save('./lib/data/normalization_statistics/training_set_feature_means',
                    zero_mean_training_data)

            np.save('./lib/data/normalization_statistics/training_set_zero_mean_feature_standard_deviations',
                    zero_mean_training_set_standard_deviations)

        # Transform training data so it has mean 0 and variance 1
        self.normalized_training_examples   = zero_mean_training_data / zero_mean_training_set_standard_deviations

        # Perform analagous transformation on the validation and test data
        adjusted_mean_validation_data       = self.validation_examples - training_set_means
        adjusted_mean_test_data             = self.test_examples       - training_set_means
        self.normalized_validation_examples = adjusted_mean_validation_data / zero_mean_training_set_standard_deviations
        self.normalized_test_examples       = adjusted_mean_test_data       / zero_mean_training_set_standard_deviations

        return self
