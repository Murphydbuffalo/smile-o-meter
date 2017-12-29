import numpy as np

class BackwardProp:
    def __init__(self, Z, A, labels):
        self.Z              = Z      # Z (W * X + B) and A (activation(Z)) are lists with L (# of layers) elements:
        self.A              = A      # [ hidden = 5 x m, hidden = 3 x m, output = 1 x m ]
        self.labels         = labels # 1 x m
        self.softmax_output = A[-1]  # 1 x m ...only the predictions for the class identified by the label of each example.

    def run(self):
        # these should have the following shapes (a derivative for each weight/bias)
        weight_gradients = [] # [5 x 2304, 3 x 5, 3 x 3]
        bias_gradients   = [] # [5 x 1,    3 x 1, 3 x 1]

        for i in range(1, len(self.A)):
            if i == 1:
                dw = self.__d_cost_d_softmax() * self.__d_softmax_d_z() * self.__d_z_d_w(-i)
                db = self.__d_cost_d_softmax() * self.__d_softmax_d_z() * self.__d_z_d_b()
            else:
                dw = weight_gradients[-1] * self.__d_relu_d_z(-i) * self.__d_z_d_w(-i)
                db = bias_gradients[-1]   * self.__d_relu_d_z(-i) * self.__d_z_d_b()

            weight_gradients.append(dw)
            bias_gradients.append(db)

        return [weight_gradients.reverse(), bias_gradients.reverse()]

    def __d_cost_d_softmax(self):
        return -1 / self.softmax_output # 1 x m

    def __d_softmax_d_z(self):
        return self.softmax_output * (1 - self.softmax_output) # 1 x m

    def __d_relu_d_z(self, layer):
        # Z[-1] ->    1 x m
        # Z[-2] ->    3 x m
        # Z[-3] ->    5 x m
        return np.where(self.Z[layer] > 0, 1, 0)

    def __d_z_d_w(self, layer):
        # A[-1] ->    1 x m
        # A[-2] ->    3 x m
        # A[-3] ->    5 x m
        return self.A[layer - 1]

    def __d_z_d_b(self):
        return 1.0
