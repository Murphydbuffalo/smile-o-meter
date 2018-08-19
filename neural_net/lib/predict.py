import numpy as np

from lib.forward_prop import ForwardProp
from lib.cost         import Cost

class Predict:
    regularization_strength = 0

    def __init__(self, examples, labels, weights, biases):
        self.examples       = examples
        self.labels         = labels
        self.weights        = weights
        self.biases         = biases
        self.num_examples   = labels.shape[1]
        self.actual_classes = labels.argmax(axis=0)

    def run(self):
        forward_prop = ForwardProp(self.weights, self.biases, self.examples)
        forward_prop.run()
        self.network_output = forward_prop.network_output

        self.cost = Cost(self.network_output,
                         self.labels,
                         self.weights,
                         self.regularization_strength).cross_entropy_loss()

        self.num_correct     = (self.actual_classes == self.predicted_classes()).sum()
        self.percent_correct = self.num_correct / self.num_examples

    def predicted_classes(self):
        return self.predictions().argmax(axis=0)

    def predictions(self):
        return self.bool_to_int(self.network_output == np.max(self.network_output, axis=0))

    def bool_to_int(self, array):
        return array * 1

    def prediction_class_percentages(self):
        return self.predictions().sum(axis=1) / self.num_examples

    def actual_class_percentages(self):
        return self.labels.sum(axis=1) / self.num_examples

    def percentage_diffs(self):
        return abs(self.prediction_class_percentages() - self.actual_class_percentages())
