import numpy as np

class Cost:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels      = labels[0]

    def cross_entropy_loss(self):
        m     = len(self.labels)
        total = 0

        for i in range(0, m):
            label       = self.labels[i]
            confidence  = self.predictions[label][i]
            total      += -np.log(confidence)

        return total / m

    # TODO: See if one of these performs better:
    # totals = [-np.log(self.predictions[self.labels[i]][i]) for i in range(m)]
    # return np.sum(totals) / m
    #
    # Haven't tested this out...
    # a = np.zip(labels, predictions)
    # np.apply_along_axis(self.__loss_for_example, 0, a)
