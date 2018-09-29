import numpy      as np
import scipy.misc as spm

class ImageRenderer:
    def render(self, pixels):
        spm.toimage(pixels).show()

    def render_48_by_48(self, pixels):
        reshaped = pixels.reshape((48, 48))
        self.render(reshaped)
