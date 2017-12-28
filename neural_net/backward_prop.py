import numpy as np

class BackwardProp:
    def __init__(self, Z, A, labels):
        self.Z              = Z
        self.A              = A
        self.labels         = labels
        self.softmax_output = np.choose(labels, A[-1])

    def run(self):
        weight_gradients = [self.__d_cost_d_w()]
        bias_gradients   = [self.__d_cost_d_b()]
        reverse_index    = -2

        for i in range(len(self.A) - 1):
            d_w = weight_gradients[-1] * self.__d_relu_d_z(reverse_index) * self.__d_z_d_w(reverse_index)
            d_b = bias_gradients[-1]   * self.__d_relu_d_z(reverse_index) * self.__d_z_d_b()

            weight_gradients.append(d_w)
            bias_gradients.append(d_b)
            reverse_index -= 1

        return [weight_gradients.reverse(), bias_gradients.reverse()]

    def __d_cost_d_w(self):
        return self.__d_cost_d_softmax() * self.__d_softmax_d_z() * self.__d_z_d_w(-1)

    def __d_cost_d_b(self):
        return self.__d_cost_d_softmax() * self.__d_softmax_d_z() * self.__d_z_d_b()

    def __d_cost_d_softmax(self):
        return -1 / self.softmax_output

    def __d_softmax_d_z(self):
        return self.softmax_output * (1 - self.softmax_output)

    def __d_relu_d_z(self, layer):
        return np.where(self.Z[layer] > 0, 1, 0)

    def __d_z_d_w(self, layer):
        return self.A[layer - 1]

    def __d_z_d_b(self):
        return 1.0
