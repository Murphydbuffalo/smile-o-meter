import numpy as np
import csv
import os

class FER_CSV:
    filename = './neural_net/data/sources/fer2013.csv'

    def __init__(self):
        self.csv    = csv.DictReader(open(self.filename))
        self.Xtrain = []
        self.Ytrain = []
        self.Xdev   = []
        self.Ydev   = []
        self.Xtest  = []
        self.Ytest  = []


    def __label(self, row):
        # For the Smile-O-Meter we want three classes: 0 = Netural, 1 = Happy,   2 = Sad
        # Need to convert from the FER labels of:      0 = Angry,   1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, 6 = Neutral
        fer_label = int(row['emotion'])

        if fer_label == 3:
            label = 1
        elif fer_label < 5:
            label = 2
        else:
            label = 0

        return label

    def __pixels(self, row):
        return np.array(row['pixels'].split(), 'int')

    def __add_data(self, label, pixels, partition):
        if partition == 'Training':
            self.Xtrain.append(pixels)
            self.Ytrain.append(label)
        elif partition == 'PrivateTest':
            self.Xdev.append(pixels)
            self.Ydev.append(label)
        elif partition == 'PublicTest':
            self.Xtest.append(pixels)
            self.Ytest.append(label)


    def load_data(self, print_progress = False):
        for index, row in enumerate(self.csv):
            partition = row['Usage']
            label     = self.__label(row)
            pixels    = self.__pixels(row)

            if print_progress == True and index % 100 == 0:
                print("partition is", partition)
                print("label is", label)
                print("shape of pixels is", pixels.shape)
                print("pixels are", pixels)

            self.__add_data(label, pixels, partition)


        self.Xtrain = np.array(self.Xtrain).T
        self.Ytrain = np.array([self.Ytrain])

        self.Xdev   = np.array(self.Xdev).T
        self.Ydev   = np.array([self.Ydev])

        self.Xtest  = np.array(self.Xtest).T
        self.Ytest  = np.array([self.Ytest])

        return True
