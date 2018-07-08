import numpy as np
from imgaug import augmenters as ia

class Augment:
    def __init__(self, images):
        self.images = images
        self.m      = images.shape[1]

    def augment(self):
        formatted_images = self.format_for_augmentation()
        augmenter        = ia.Sequential([
            # horizontally flip 100% of the images
            ia.Fliplr(1),
            # translate images by 0-25%
            ia.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
        ])

        augmented = augmenter.augment_images(formatted_images)

        return self.format_for_training(augmented)

    def format_for_augmentation(self):
        reshaped = self.images.T.reshape((self.m, 48, 48, 1))

        return reshaped.astype(np.uint8)

    def format_for_training(self):
        return self.augmented.reshape((self.m, 2304)).T
