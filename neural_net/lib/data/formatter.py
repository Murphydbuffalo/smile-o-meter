import numpy as np
from lib.data.augment import Augment

class Formatter:
    def __init__(self, data):
        self.data         = data
        self.Xtrain       = data.Xtrain
        self.Ytrain       = data.Ytrain
        self.Xtest        = data.Xtest
        self.Ytest        = data.Ytest
        self.num_features = self.Xtrain.shape[0]
        self.num_classes  = self.Ytrain.shape[0]

    def run(self):
        self.__remove_bad_data()
        self.__augment()
        self.__normalize()

        return self

    def __remove_bad_data(self):
        Xtrain, Ytrain = self.__remove_zero_standard_deviation_data(self.Xtrain, self.Ytrain)
        self.Xtrain    = Xtrain
        self.Ytrain    = Ytrain

        Xtest, Ytest = self.__remove_zero_standard_deviation_data(self.Xtest, self.Ytest)
        self.Xtest   = Xtest
        self.Ytest   = Ytest

    def __augment(self):
        additional_training_examples = Augment(self.Xtrain).augment()

        self.Xtrain = np.column_stack((self.Xtrain, additional_training_examples))
        self.Ytrain = np.column_stack((self.Ytrain, self.Ytrain))

    def __normalize(self):
        self.training_set_means                         = np.array([np.mean(self.Xtrain, 1)]).T
        zero_mean_training_data                         = self.Xtrain - self.training_set_means
        adjusted_mean_test_data                         = self.Xtest  - self.training_set_means
        self.zero_mean_training_set_standard_deviations = np.array([np.std(zero_mean_training_data, 1)]).T

        # Transform training data so it has mean 0 and variance 1, apply identical transformation to test data
        self.Xtrain_norm = (zero_mean_training_data) / self.zero_mean_training_set_standard_deviations
        self.Xtest_norm  = (adjusted_mean_test_data) / self.zero_mean_training_set_standard_deviations

    def __remove_zero_standard_deviation_data(self, matrix, labels):
        standard_deviations                   = np.std(matrix, 0)
        columns_with_zero_standard_devivation = np.where(standard_deviations == 0)[0]

        return [
            np.delete(matrix, columns_with_zero_standard_devivation, axis = 1),
            np.delete(labels, columns_with_zero_standard_devivation, axis = 1)
        ]
