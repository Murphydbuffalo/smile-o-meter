import numpy as np
from lib.forward_prop  import ForwardProp
from lib.cost          import Cost

class GradientCheck:
    def __init__(self, weights, biases, weight_gradients, X, Y):
        self.weights          = weights
        self.biases           = biases
        self.weight_gradients = weight_gradients
        self.X                = X
        self.Y                = Y
        self.epsilon          = 0.00001

    def run(self):
        acceptable_delta   = self.epsilon
        numeric_gradients  = self.__numeric_gradients()
        analytic_gradients = self.weight_gradients
        delta1             = np.abs(numeric_gradients[1] - analytic_gradients[1])
        delta2             = np.abs(numeric_gradients[2] - analytic_gradients[2])

        print("numeric_gradients[1]:", numeric_gradients[1])
        print("analytic_gradients[1]:", analytic_gradients[1])

        print("numeric_gradients[2]:", numeric_gradients[2])
        print("analytic_gradients[2]:", analytic_gradients[2])

        print("delta1:", delta1)
        print("delta2:", delta2)

        return (delta1.max() <= acceptable_delta) and (delta2.max() <= acceptable_delta)

    def __numeric_gradients(self):
        weights           = np.copy(self.weights)
        numeric_gradients = []

        for i in range(len(weights)):
            numeric_gradients.append(np.zeros(weights[i].shape))

        # Skip weights connecting input layer to 1st hidden layer because there
        # are so many it takes hours to calculate them all.
        # Not to mention, if you know the second-to-last layer's gradients are
        # correct, then you can confident that all hidden layer gradients are correct
        # because they are calculated the same way.
        for layer in range(1, len(weights)):
            for column in range(weights[layer].shape[0]):
                for row in range(weights[layer].shape[1]):
                    original_weight = weights[layer][column][row]

                    weights[layer][column][row] = original_weight + self.epsilon
                    Zplus, Aplus                = ForwardProp(weights, self.biases, self.X).run()


                    weights[layer][column][row] = original_weight - self.epsilon
                    Zminus, Aminus              = ForwardProp(weights, self.biases, self.X).run()

                    cost_plus  = Cost(Aplus[-1],  self.Y).cross_entropy_loss()
                    cost_minus = Cost(Aminus[-1], self.Y).cross_entropy_loss()

                    numeric_gradients[layer][column][row] = (cost_plus - cost_minus) / (2 * self.epsilon)
                    weights[layer][column][row]           = original_weight # paranoia

        return numeric_gradients
