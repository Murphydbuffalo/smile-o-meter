import numpy as np

class Cost:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels      = labels[0]

    def cross_entropy_loss(self):
        m     = len(self.labels)
        total = 0
        # totals = [-np.log(self.predictions[self.labels[i]][i]) for i in range(m)]
        # return np.sum(totals) / m
        for i in range(0, m):
            label       = self.labels[i]
            confidence  = self.predictions[label][i]
            total      += -np.log(confidence)

        return total / m

    # label       =         1
    # predictions = [0.12, 0.80, 0.08]
    # -np.log(predictions[label])
    # cost = np.sum([
    #     -np.log(0.12),
    #     -np.log(0.80),
    #     -np.log(0.08)
    # ])
