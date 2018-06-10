import numpy as np
import csv
import os

class FER_CSV:
    filename = './lib/data/sources/fer2013.csv'

    def __init__(self):
        self.csv    = csv.DictReader(open(self.filename))
        self.Xtrain = []
        self.Ytrain = np.array([[],[],[]]) # Our Y matrices will have shape 3 x m (we have 3 classes)
        self.Xtest  = []
        self.Ytest  = np.array([[],[],[]])


    def __label(self, row):
        # For the Smile-O-Meter we want three classes: 0 = Netural, 1 = Happy,     2 = Sad
        # Need to convert from the FER labels of:      0 = Angry,   1 = Disgusted, 2 = Afraid, 3 = Happy, 4 = Sad, 5 = Surprised, 6 = Neutral
        fer_label = int(row['emotion'])

        if fer_label == 6:
            label = np.array([[1],[0],[0]])
        elif fer_label == 3:
            label = np.array([[0],[1],[0]])
        else:
            label = np.array([[0],[0],[1]])

        return label

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
