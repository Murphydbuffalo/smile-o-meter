import numpy as np

class Cost:
    def __init__(self, predictions, labels, weights, lambda):
        self.predictions = predictions
        self.labels      = labels
        self.m           = float(labels.shape[1])
        self.weights     = weights
        self.lambda      = lambda

    def cross_entropy_loss(self):
        return (np.sum(-np.log(self.predictions) * self.labels) / self.m) + self.l2_regularization_loss()

    def l2_regularization_loss(self):
        return 0.5 * self.lambda * np.square(self.weights).sum()
