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
                d_cost_d_softmax_z = self.__d_cost_d_softmax() * self.__d_softmax_d_z() # 3 x m
                d_z_d_a            = self.__d_z_d_a(-i).dot(d_cost_d_softmax_z)         # 3 x 3 * 3 x m = 3 x m
                dw                 = d_cost_d_softmax_z.dot(self.__d_z_d_w(-i).T)       # 3 x m * m x 3 = 3 x 3
                db                 = np.sum(d_cost_d_softmax_z, axis = 1, keepdims = True) * self.__d_z_d_b() # 3 x 1
            else:
                d_relu_d_z = self.__d_relu_d_z(-i) * d_z_d_a      # 3 x m elementwise* 3 x m = 3 x m for layer 2
                dw         = d_relu_d_z.dot(self.__d_z_d_w(-i).T) # 3 x m * m x 5 = 3 x 5 for layer 2
                db         = np.sum(d_relu_d_z, axis = 1, keepdims = True) * self.__d_z_d_b()
                d_z_d_a    = self.__d_z_d_a(-i).T.dot(d_relu_d_z) # 5 x 3 *  3 x m =  5 x m

            weight_gradients.append(dw)
            bias_gradients.append(db)

        return [list(reversed(weight_gradients)), list(reversed(bias_gradients))]

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

    # Partial derivative of a layer's Z with respect to the previous layer's A
    def __d_z_d_a(self, layer):
        # W[-1] ->    3 x 3
        # W[-2] ->    3 x 5
        # W[-3] ->    5 x 2304
        return self.weights[layer]

    def __d_z_d_b(self):
        return 1.0
