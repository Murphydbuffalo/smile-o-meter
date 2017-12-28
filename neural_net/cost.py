import numpy as np

class Cost:
    def __init__(self, predictions, labels):
        # take only the softmax output for the class of the label
        self.predictions = np.choose(labels, predictions)
        self.labels      = labels
        self.m           = float(labels.shape[1])

    def cross_entropy_loss(self):
        return np.sum(-np.log(self.predictions)) / self.m
