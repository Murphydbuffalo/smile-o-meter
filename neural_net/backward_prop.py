import numpy as np

class BackwardProp:
    def __init__(self, weights, Z, A, labels):
        self.weights        = weights # [ 5 x 2304, 3 x 5, 3 x 3 ]
        self.Z              = Z       # Z = W_current_layer * A_previous_layer + B_current_layer and A = activation(Z) are lists with L (# of layers) elements:
        self.A              = A       # [ 5 x m (hidden), 3 x m (hidden), 3 x m (output) ]
        self.labels         = labels  # 1 x m
        self.softmax_output = A[-1]   # 1 x m

    def run(self):
        # these should have the following shapes (a derivative for each weight/bias)
        weight_gradients = [] # [5 x 2304, 3 x 5, 3 x 3]
        bias_gradients   = [] # [5 x    1, 3 x 1, 3 x 1]

        for i in range(1, len(self.A)):
            if i == 1:
                dw = np.dot(self.__d_cost_d_softmax() * self.__d_softmax_d_z(), self.__d_z_d_w(-i).T)
                db = np.sum(self.__d_cost_d_softmax() * self.__d_softmax_d_z(), axis = 1, keepdims = True) * self.__d_z_d_b()
            else:
                dw = np.dot(weight_gradients[-1], np.dot(self.__d_z_d_w(-i), self.__d_relu_d_z(-i).T).T)
                db = np.sum(self.__d_relu_d_z(-i), axis = 1, keepdims = True) * bias_gradients[-1] * self.__d_z_d_b()

            weight_gradients.append(dw)
            bias_gradients.append(db)

        return [weight_gradients.reverse(), bias_gradients.reverse()]

    # TODO: Fix this. Need to define softmax as a function that takes both Z and
    # Y as inputs, and the derivative should reflect this.
    def __d_cost_d_softmax(self):
        return -1 / self.softmax_output # 3 x m

    def __d_softmax_d_z(self):
        return self.softmax_output * (1 - self.softmax_output) # 3 x m

    def __d_relu_d_z(self, layer):
        # No ReLU in the first layer
        # Z[-2] ->    3 x m
        # Z[-3] ->    5 x m
        return np.where(self.Z[layer] > 0, 1, 0)

    def __d_z_d_w(self, layer):
        # A[-1] ->       3 x m
        # A[-2] ->       3 x m
        # A[-3] ->       5 x m
        # A[-4] ->    2304 x m
        return self.A[layer - 1]

    def __d_z_d_b(self):
        return 1.0
