import numpy as np
import csv
import os

class FER_CSV:
    filename = './lib/data/sources/fer2013.csv'

    def __init__(self):
        self.csv               = csv.DictReader(open(self.filename))
        self.training_examples = []
        self.test_examples     = []
        self.training_labels   = self.k_dimensional_array()
        self.test_labels       = self.k_dimensional_array()

    def load_data(self, print_progress = False):
        for index, row in enumerate(self.csv):
            partition = row['Usage']
            label     = self.label(row)
            pixels    = self.pixels(row)

            if print_progress == True and index % 1000 == 0:
                print("partition is", partition)
                print("label is", label)
                print("shape of pixels is", pixels.shape)
                print("pixels are", pixels)

            self.add_data(label, pixels, partition)


        self.training_examples = np.array(self.training_examples).T
        self.test_examples     = np.array(self.test_examples).T

        return self

    # Where `k` is the number of classes in our classifier/data set.
    def k_dimensional_array(self):
        return np.array([[], [], [], [], [], [], []])

    # Convert an integer into a "one-hot" vector of 0s and 1s, with the sole 1 at
    # the index corresponding to the integer.
    def label(self, row):
        one_hot_vector            = np.zeros((7, 1))
        fer_label                 = int(row['emotion'])
        one_hot_vector[fer_label] = 1

        return one_hot_vector

    def pixels(self, row):
        return np.array(row['pixels'].split(), 'int')

    # TODO: REFACTOR
    # Can you build these arrays in a cleaner way via better use of Numpy?
    # Or at least build them in a consistent way instead of using both `append`
    # and `column_stack`?
    def add_data(self, label, pixels, partition):
        if partition == 'PublicTest':
            self.test_examples.append(pixels)
            self.test_labels = np.column_stack((self.test_labels, label))
        else:
            self.training_examples.append(pixels)
            self.training_labels = np.column_stack((self.training_labels, label))
