import numpy as np
import csv
import os

class FER_CSV:
    filename = './lib/data/sources/fer2013.csv'

    def __init__(self):
        self.csv    = csv.DictReader(open(self.filename))
        self.Xtrain = []
        self.Xtest  = []
        self.Ytrain = self.__k_dimensional_array()
        self.Ytest  = self.__k_dimensional_array()


    # Where `k` is the number of classes in our classifier/data set.
    def __k_dimensional_array(self):
        return np.array([[], [], [], [], [], [], []])

    # Convert an integer into a "one-hot" vector of 0s and 1s, with the sole 1 at
    # the index corresponding to the integer.
    def __label(self, row):
        one_hot_vector            = np.zeros((7, 1))
        fer_label                 = int(row['emotion'])
        one_hot_vector[fer_label] = 1

        return one_hot_vector

    def __pixels(self, row):
        return np.array(row['pixels'].split(), 'int')

    def __add_data(self, label, pixels, partition):
        if partition == 'PublicTest':
            self.Xtest.append(pixels)
            self.Ytest = np.column_stack((self.Ytest, label))
        else:
            self.Xtrain.append(pixels)
            self.Ytrain = np.column_stack((self.Ytrain, label))

    def load_data(self, print_progress = False):
        for index, row in enumerate(self.csv):
            partition = row['Usage']
            label     = self.__label(row)
            pixels    = self.__pixels(row)

            if print_progress == True and index % 1000 == 0:
                print("partition is", partition)
                print("label is", label)
                print("shape of pixels is", pixels.shape)
                print("pixels are", pixels)

            self.__add_data(label, pixels, partition)


        self.Xtrain = np.array(self.Xtrain).T
        self.Xtest  = np.array(self.Xtest).T

        return True
