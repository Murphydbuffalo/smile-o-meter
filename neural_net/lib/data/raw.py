import numpy as np
import csv
import os

# We're using the Facial Expression Recognition ("FER") dataset from Kaggle:
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
class Raw:
    num_classes = 7

    def __init__(self, filename = './lib/data/sources/fer2013.csv'):
        self.file     = open(filename)
        self.csv      = csv.DictReader(self.file)
        self.datasets = {
            'Training':    { 'examples': [], 'labels': [] },
            'PrivateTest': { 'examples': [], 'labels': [] },
            'PublicTest':  { 'examples': [], 'labels': [] }
        }

    def load(self):
        try:
            for row in self.csv:
                pixels  = row['pixels'].split()
                if self.bad_data(pixels): continue

                dataset = row['Usage']
                label   = self.label(row)

                self.datasets[dataset]['examples'].append(pixels)
                self.datasets[dataset]['labels'].append(label)

            self.training_examples = self.to_numpy_array(self.datasets['Training']['examples'])
            self.training_labels   = self.flatten_labels(
                self.to_numpy_array(self.datasets['Training']['labels'])
            )

            self.validation_examples = self.to_numpy_array(self.datasets['PrivateTest']['examples'])
            self.validation_labels   = self.flatten_labels(
                self.to_numpy_array(self.datasets['PrivateTest']['labels'])
            )

            self.test_examples = self.to_numpy_array(self.datasets['PublicTest']['examples'])
            self.test_labels   = self.flatten_labels(
                self.to_numpy_array(self.datasets['PublicTest']['labels'])
            )

            return self
        finally:
            self.file.close()

    # Convert an integer into a "one-hot" vector of 0s and 1s, with the sole 1
    # at the index corresponding to the integer. Eg, with 7 possible classes the
    # zero-indexed label `3` becomes `[0, 0, 0, 1, 0, 0, 0]`. The neural network
    # outputs its predictions as vectors of this form, so it's easiest to convert
    # the labels to the same form for comparison.
    def label(self, row):
        one_hot_vector            = [[0]] * self.num_classes
        fer_label                 = int(row['emotion'])
        one_hot_vector[fer_label] = [1]

        return one_hot_vector

    def to_numpy_array(self, list):
        return np.array(list, 'uint8').T

    def flatten_labels(self, labels):
        return labels.reshape((self.num_classes, -1))

    def bad_data(self, pixels):
        array = np.array(pixels, 'uint8')
        return array.std() == 0
