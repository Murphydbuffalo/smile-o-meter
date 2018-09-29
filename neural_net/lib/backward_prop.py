import numpy as np

class BackwardProp:
    def __init__(self, weights, Z, A, labels, regularization_strength):
        self.weights                 = weights
        self.labels                  = labels
        self.num_examples            = labels.shape[1]
        self.regularization_strength = regularization_strength
        self.Z                       = Z # Z = W_current_layer * A_previous_layer + B_current_layer
        self.A                       = A # A = activation(Z) where activation is ReLU for hidden layers and Softmax for the output layer
        self.softmax_output          = A[-1]

    def run(self):
        # These should have a derivative for each weight and bias
        weight_gradients = []
        bias_gradients   = []

        for i in range(1, len(self.A)):
            if i == 1:
                dz = self.d_cost_d_z()
            else:
                dz = self.d_z_d_a(-i).T.dot(dz) * self.d_relu_d_z(-i)

            dw = dz.dot(self.d_z_d_w(-i).T) + self.d_regularization(-i)
            db = np.sum(dz, axis = 1, keepdims = True) * self.d_z_d_b()

            weight_gradients.append(dw)
            bias_gradients.append(db)

        return [np.array(list(reversed(weight_gradients))), np.array(list(reversed(bias_gradients)))]

    def d_cost_d_z(self):
        return (self.softmax_output - self.labels) / self.num_examples

    def d_relu_d_z(self, layer):
        return np.where(self.Z[layer] > 0, 1, 0)

    def d_z_d_w(self, layer):
        return self.A[layer - 1]

    def d_z_d_a(self, layer):
        return self.weights[layer + 1]

    def d_z_d_b(self):
        return 1.0

    def d_regularization(self, layer):
        return self.regularization_strength * self.weights[layer]
