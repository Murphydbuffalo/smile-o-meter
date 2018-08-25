import numpy as np

class Cost:
    def __init__(self, predictions, labels, weights, lambd):
        self.predictions = predictions
        self.labels      = labels
        self.m           = float(labels.shape[1])
        self.weights     = weights
        self.lambd       = lambd

    def cross_entropy_loss(self):
        inverse_log         = -np.log(self.predictions) * self.labels
        average_inverse_log = np.sum(inverse_log) / self.m

        return average_inverse_log + self.l2_regularization_loss()

    def l2_regularization_loss(self):
        squared_weights = np.square(self.weights)
        weight_sum      = 0

        for layer in range(len(squared_weights)):
            weight_sum += squared_weights[layer].sum()

        return 0.5 * self.lambd * weight_sum
