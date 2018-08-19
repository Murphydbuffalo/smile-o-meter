from sys import argv

from lib.forward_prop    import ForwardProp
from lib.cost            import Cost
from lib.backward_prop   import BackwardProp
from lib.optimizers.adam import Adam
from lib.gradient_check  import GradientCheck

class Optimize:
    regularization_strength = 0.001

    def __init__(self, examples, labels, optimizer):
        self.examples     = examples
        self.labels       = labels
        self.optimizer    = optimizer
        self.weights      = optimizer.weights
        self.biases       = optimizer.biases
        self.current_cost = 9999
        self.costs        = []
        self.logger       = OptimizationLogger(self)

    def run(self):
        while self.current_cost > 0.1:
            forward_prop = ForwardProp(self.weights, self.biases, self.examples)
            [linear_activation, nonlinear_activation] = forward_prop.run()

            self.linear_activation    = linear_activation
            self.nonlinear_activation = nonlinear_activation

            self.current_cost = self.calculate_cost(forward_prop.network_output)
            self.costs.append(self.current_cost)

            [weight_gradients, bias_gradients] = self.backward_prop()

            self.optimizer.update_parameters(weight_gradients, bias_gradients)
            self.weights = self.optimizer.weights
            self.biases  = self.optimizer.biases

            self.logger.log(weight_gradients)

        return {
            'costs':   costs,
            'weights': self.weights,
            'biases':  self.biases
        }

    def current_cost(self):
        return self.costs[-1]

    def backward_prop(self):
        return BackwardProp(self.weights,
                            self.linear_activation,
                            self.nonlinear_activation,
                            self.labels,
                            self.regularization_strength).run()

    def calculate_cost(self, network_output):
        return Cost(network_output,
                    self.labels,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

class OptimizationLogger:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.iteration = 0

    def log(self, weight_gradients):
        if (self.iteration % 10) == 0:
            print("Iteration #", self.iteration)
            print("Cost is", self.optimizer.current_cost)

        if len(argv) > 1 and argv[1] == '--check-gradients' and self.iteration % 100 == 0:
            check = GradientCheck(self.optimizer.weights,
                                  self.optimizer.biases,
                                  weight_gradients,
                                  self.optimizer.examples,
                                  self.optimizer.labels)

            print("Are the analytic gradients about the same as the numeric gradients?", check.run())

        self.iteration += 1
