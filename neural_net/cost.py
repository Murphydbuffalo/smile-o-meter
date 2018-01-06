import numpy as np

class Cost:
    def __init__(self, predictions, labels):
        # To calculate the cost of the network we take only the softmax predictions
        # for the class corresponding to the label of each example.
        self.predictions = np.choose(labels, predictions)
        self.m           = float(labels.shape[1])

    def cross_entropy_loss(self):
        return np.sum(-np.log(self.predictions)) / self.m
