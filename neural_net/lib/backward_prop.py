import numpy as np

class BackwardProp:
    def __init__(self, weights, Z, A, labels):
        self.weights        = weights # [ 5 x 2304, 3 x 5, 3 x 3 ]
        self.Z              = Z       # Z = W_current_layer * A_previous_layer + B_current_layer and A = activation(Z) are lists with L (# of layers) elements:
        self.A              = A       # [ 5 x m (hidden), 3 x m (hidden), 3 x m (output) ]
        self.labels         = labels  # 3 x m
        self.softmax_output = A[-1]   # 3 x m
        self.m              = labels.shape[1]

    def run(self):
        # these should have the following shapes (a derivative for each weight/bias)
        weight_gradients = [] # [5 x 2304, 3 x 5, 3 x 3]
        bias_gradients   = [] # [5 x    1, 3 x 1, 3 x 1]

        for i in range(1, len(self.A)):
            if i == 1:
                dz = self.__d_cost_d_z()
            else:
                dz = self.__d_z_d_a(-i).T.dot(dz) * self.__d_relu_d_z(-i)

            dw = dz.dot(self.__d_z_d_w(-i).T)
            db = np.sum(dz, axis = 1, keepdims = True) * self.__d_z_d_b()

            weight_gradients.append(dw)
            bias_gradients.append(db)

        return [np.array(list(reversed(weight_gradients))), np.array(list(reversed(bias_gradients)))]

    def __d_cost_d_z(self):
        return (self.softmax_output - self.labels) / self.m # 3 x m

    def __d_relu_d_z(self, layer):
        # No ReLU in the output layer
        # Z[-2] ->    3 x m
        # Z[-3] ->    5 x m
        return np.where(self.Z[layer] > 0, 1, 0)

    def __d_z_d_w(self, layer):
        # A[-1] ->       3 x m
        # A[-2] ->       3 x m
        # A[-3] ->       5 x m
        # A[-4] ->    2304 x m
        return self.A[layer - 1]

    def __d_z_d_a(self, layer):
        # W[-1] ->    3 x 3
        # W[-2] ->    3 x 5
        # W[-3] ->    5 x 2304
        return self.weights[layer + 1]

    def __d_z_d_b(self):
        return 1.0
