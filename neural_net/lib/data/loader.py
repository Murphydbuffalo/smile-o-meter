import numpy as np
from lib.data.sources.fer_csv import FER_CSV

class Loader:
    max_num_pixels = 2304
    data_sources   = [
        FER_CSV()
    ]

    def __init__(self, print_progress = False):
        self.print_progress = print_progress
        self.Xtrain         = None
        self.Ytrain         = None
        self.Xtest          = None
        self.Ytest          = None

    def load(self):
        for data_source in self.data_sources:
            data_source.load_data(self.print_progress)

            if self.Xtrain == None:
                self.Xtrain = data_source.Xtrain
                self.Ytrain = data_source.Ytrain

                self.Xtest = data_source.Xtest
                self.Ytest = data_source.Ytest
            else:
                self.Xtrain = np.column_stack((self.Xtrain, data_source.Xtrain))
                self.Ytrain = np.column_stack((self.Ytrain, data_source.Ytrain))

                self.Xtest = np.column_stack((self.Xtest, data_source.Xtest))
                self.Ytest = np.column_stack((self.Ytest, data_source.Ytest))

        Xtrain_no_bad_data, Ytrain_no_bad_data = self.__remove_zero_standard_deviation_examples(self.Xtrain, self.Ytrain)
        self.Xtrain = Xtrain_no_bad_data
        self.Ytrain = Ytrain_no_bad_data

        Xtest_no_bad_data, Ytest_no_bad_data = self.__remove_zero_standard_deviation_examples(self.Xtest, self.Ytest)
        self.Xtest = Xtest_no_bad_data
        self.Ytest = Ytest_no_bad_data

        self.training_set_means                         = np.array([np.mean(self.Xtrain, 1)]).T
        zero_mean_training_data                         = self.Xtrain - self.training_set_means
        adjusted_mean_test_data                         = self.Xtest  - self.training_set_means
        self.zero_mean_training_set_standard_deviations = np.array([np.std(zero_mean_training_data, 1)]).T

        # Transform training data so it has mean 0 and variance 1, apply identical transformation to test data
        self.Xtrain_norm =  (zero_mean_training_data) / self.zero_mean_training_set_standard_deviations
        self.Xtest_norm  =  (adjusted_mean_test_data) / self.zero_mean_training_set_standard_deviations

        # Descriptive names
        self.num_features = self.Xtrain.shape[0]
        self.num_classes  = self.Ytrain.shape[0]

        return self

    def __remove_zero_standard_deviation_examples(self, matrix, labels):
        standard_deviations                   = np.std(matrix, 0)
        columns_with_zero_standard_devivation = np.where(standard_deviations == 0)[0]

        return [
            np.delete(matrix, columns_with_zero_standard_devivation, axis = 1),
            np.delete(labels, columns_with_zero_standard_devivation, axis = 1)
        ]
