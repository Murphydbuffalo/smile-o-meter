import numpy as np
from data.sources.fer_csv import FER_CSV

class Loader:
    max_num_pixels = 2304
    data_sources   = [
        FER_CSV()
    ]

    def __init__(self, print_progress = False):
        self.print_progress = print_progress
        self.Xtrain         = None
        self.Ytrain         = None
        self.Xdev           = None
        self.Ydev           = None
        self.Xtest          = None
        self.Ytest          = None

    # Normalizes input data to have mean 0 and variance 1
    def __norm(self, matrix):
        means                                 = np.mean(matrix, 0)
        mean_zero_data                        = matrix - means
        standard_deviations                   = np.std(mean_zero_data, 0)
        normalized                            = mean_zero_data / standard_deviations
        columns_with_zero_standard_devivation = np.where(standard_deviations == 0)[0]

        return np.delete(normalized, columns_with_zero_standard_devivation, axis = 1)

    def normalize(self):
        self.Xtrain_norm = self.__norm(self.Xtrain)
        self.Xdev_norm   = self.__norm(self.Xdev)
        self.Xtest_norm  = self.__norm(self.Xtest)

    def load(self):
        for data_source in self.data_sources:
            data_source.load_data(self.print_progress)

            if self.Xtrain == None:
                self.Xtrain = data_source.Xtrain
                self.Ytrain = data_source.Ytrain

                self.Xdev = data_source.Xdev
                self.Ydev = data_source.Ydev

                self.Xtest = data_source.Xtest
                self.Ytest = data_source.Ytest
            else:
                self.Xtrain = np.column_stack((self.Xtrain, data_source.Xtrain))
                self.Ytrain = np.column_stack((self.Ytrain, data_source.Ytrain))

                self.Xdev = np.column_stack((self.Xdev, data_source.Xdev))
                self.Ydev = np.column_stack((self.Ydev, data_source.Ydev))

                self.Xtest = np.column_stack((self.Xtest, data_source.Xtest))
                self.Ytest = np.column_stack((self.Ytest, data_source.Ytest))
