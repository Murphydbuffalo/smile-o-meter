import numpy as np

from imgaug import augmenters

class Augmenter:
    def __init__(self, images, labels):
        self.images                          = images
        self.labels                          = labels
        self.num_features, self.num_examples = images.shape
        self.pixel_dimension                 = int(np.sqrt(self.num_features))

        # See https://imgaug.readthedocs.io/en/latest/source/augmenters.html
        self.transformations = [
            # Flip horizontally
            [augmenters.Fliplr(1)],
            # Change brightness via multiplication
            [augmenters.Multiply((0.5, 1.5))],
            # Rotate up to 45 degrees in either direction
            [augmenters.Affine(rotate=(-45, 45))],
        ]
        self.num_transformations = len(self.transformations)

    def augment(self):
        results = self.format_for_augmentation(self.images)

        for transformation in self.transformations:
            formatted_images = self.format_for_augmentation(self.images)
            augmenter        = augmenters.Sequential(transformation)
            augmented        = augmenter.augment_images(formatted_images)
            results          = np.concatenate((results, augmented), axis=0)

        # Repeat the labels once for each augmented set of images
        self.augmented_labels   = np.tile(self.labels, self.num_transformations + 1)
        self.augmented_examples = self.format_for_training(results)

        return self

    def format_for_augmentation(self, original_images):
        # ImgAug expects images formatted as a 4D numpy array of shape
        # `(n, height, width, rgb_channels)`. We use greyscale images, for which
        # ImgAug expects only a single channel.
        return original_images.T.reshape(
            (self.num_examples, self.pixel_dimension, self.pixel_dimension, 1)
        )

    def format_for_training(self, augmented_images):
        augmented_num_examples = (self.num_transformations + 1) * self.num_examples
        return np.array(augmented_images).reshape((augmented_num_examples, self.num_features)).T
