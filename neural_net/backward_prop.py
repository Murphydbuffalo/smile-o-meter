import numpy as np

class BackwardProp:
    def __init__(self, Z, A, label, cost, weights, biases):
        self.Z              = Z
        self.A              = A
        self.label          = label
        self.cost           = cost
        self.weights        = weights
        self.biases         = biases
        self.softmax_output = A[-1][label]

    def run(self):
        weight_gradients = [self.__d_cost_d_w()]
        bias_gradients   = [self.__d_cost_d_b()]
        reverse_index    = -1

        for i in range(len(weights)):
            previous_weight_gradients = weight_gradients[reverse_index]
            previous_bias_gradients   = bias_gradients[reverse_index]
            reverse_index            -= 1

            d_w = previous_weight_gradients * self.__d_relu_d_z(reverse_index) * self.__d_z_d_w(reverse_index)
            d_b = previous_bias_gradients   * self.__d_relu_d_z(reverse_index) * self.__d_z_d_b()

            weight_gradients.append(d_w)
            bias_gradients.append(d_b)

        return [weight_gradients, bias_gradients]

    def __d_cost_d_w(self):
        return self.__d_cost_d_softmax() * self.__d_softmax_d_z() * self.__d_z_d_w(-1)

    def __d_cost_d_b(self):
        return self.__d_cost_d_softmax() * self.__d_softmax_d_z() * self.__d_z_d_b()

    def __d_cost_d_softmax(self):
        return -1 / self.softmax_output

    def __d_softmax_d_z(self):
        return self.softmax_output * (1 - self.softmax_output)

    def __d_relu_d_z(self, layer):
        np.where(self.Z[layer] > 0, 1, 0)

    def __d_z_d_w(self, layer):
        return self.A[layer - 1]

    def __d_z_d_b(self):
        return 1.0
