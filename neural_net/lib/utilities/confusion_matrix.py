from sklearn.metrics import confusion_matrix

class ConfusionMatrix:
    def __init__(self, actual_classes, predicted_classes):
        self.actual_classes    = actual_classes
        self.predicted_classes = predicted_classes

    def string(self):
        return str(self.array())

    def array(self):
        return confusion_matrix(self.actual_classes, self.predicted_classes)
