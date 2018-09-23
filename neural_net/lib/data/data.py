import numpy as np

from lib.data.raw        import Raw
from lib.data.augmenter  import Augmenter
from lib.data.normalizer import Normalizer

class Data:
    def __init__(self, filename = './lib/data/sources/fer2013.csv'):
        self.filename = filename

    def build(self):
        raw_data   = Raw(self.filename).load()
        augmenter  = Augmenter(raw_data.training_examples,
                               raw_data.training_labels).augment()

        normalizer = Normalizer(augmenter.augmented_examples,
                                raw_data.validation_examples,
                                raw_data.test_examples).normalize()

        self.training_examples = normalizer.normalized_training_examples
        self.training_labels   = augmenter.augmented_labels

        self.validation_examples = normalizer.normalized_validation_examples
        self.validation_labels   = raw_data.validation_labels

        self.test_examples = normalizer.normalized_test_examples
        self.test_labels   = raw_data.test_labels

        return self

    def save(self):
        np.save('./lib/data/sources/training_examples',   self.training_examples)
        np.save('./lib/data/sources/training_labels',     self.training_labels)
        np.save('./lib/data/sources/validation_examples', self.validation_examples)
        np.save('./lib/data/sources/validation_labels',   self.validation_labels)
        np.save('./lib/data/sources/test_examples',       self.test_examples)
        np.save('./lib/data/sources/test_labels',         self.test_labels)

        return self

    def load(self):
        self.training_examples   = np.load('./lib/data/sources/training_examples.npy')
        self.training_labels     = np.load('./lib/data/sources/training_labels.npy')
        self.validation_examples = np.load('./lib/data/sources/validation_examples.npy')
        self.validation_labels   = np.load('./lib/data/sources/validation_labels.npy')
        self.test_examples       = np.load('./lib/data/sources/test_examples.npy')
        self.test_labels         = np.load('./lib/data/sources/test_labels.npy')

        return self
