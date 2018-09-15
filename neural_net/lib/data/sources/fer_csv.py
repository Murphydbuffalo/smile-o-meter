import numpy as np
import csv
import os

# FER = "Facial Expression Recognition"
# See https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
class FER_CSV:
    def __init__(self, filename = './lib/data/sources/fer2013.csv'):
        self.file = open(filename)
        self.csv  = csv.DictReader(self.file)
        self.data = {
            'Training':    { 'examples': [], 'labels': [] },
            'PublicTest':  { 'examples': [], 'labels': [] },
            'PrivateTest': { 'examples': [], 'labels': [] }
        }

    def load_data(self):
        try:
            for row in self.csv:
                data_set = row['Usage']
                pixels   = row['pixels'].split()
                label    = self.label(row)

                self.data[data_set]['examples'].append(pixels)
                self.data[data_set]['labels'].append(label)

            self.training_examples   = self.array(self.data['Training']['examples'])
            self.training_labels     = self.array(self.data['Training']['labels']).reshape((7, -1))
            self.validation_examples = self.array(self.data['PublicTest']['examples'])
            self.validation_labels   = self.array(self.data['PublicTest']['labels']).reshape((7, -1))
            self.test_examples       = self.array(self.data['PrivateTest']['examples'])
            self.test_labels         = self.array(self.data['PrivateTest']['labels']).reshape((7, -1))

            return self
        finally:
            self.file.close()

    # Convert an integer into a "one-hot" vector of 0s and 1s, with the sole 1
    # at the index corresponding to the integer. Eg, with 7 possible classes the
    # zero-indexed label `3` becomes `[0, 0, 0, 1, 0, 0, 0]`. The neural network
    # outputs its predictions as vectors of this form, so it's easiest to convert
    # the labels to the same form for comparison.
    def label(self, row):
        one_hot_vector            = [[0],[0],[0],[0],[0],[0],[0]]
        fer_label                 = int(row['emotion'])
        one_hot_vector[fer_label] = [1]

        return one_hot_vector

    def array(self, list):
        return np.array(list, 'uint8').T
