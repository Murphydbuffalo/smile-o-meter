import numpy as np

class Cost:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels      = labels
        self.m           = float(labels.shape[1])

    # To calculate the cost of the network we take only the softmax predictions
    # for the class corresponding to the label of each example.
    def cross_entropy_loss(self):
        near_zero                                      = 10 ** -300
        self.predictions[self.predictions < near_zero] = near_zero

        return np.sum(-np.log(self.predictions) * self.labels) / self.m
