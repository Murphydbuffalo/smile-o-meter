import numpy      as np
import scipy.misc as spm

class Renderer:
    def render(self, pixels):
        spm.toimage(pixels).show()

    def render48by48(self, pixels):
        reshaped = pixels.reshape((48, 48))
        self.render(reshaped)
