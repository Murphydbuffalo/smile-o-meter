import numpy as np

from lib.forward_prop    import ForwardProp
from lib.cost            import Cost
from lib.backward_prop   import BackwardProp

class Optimize:
    def __init__(self, examples, labels, optimizer, regularization_strength, num_epochs = 1000, batch_size = 128, logging_enabled = True):
        self.examples                = examples
        self.labels                  = labels
        self.optimizer               = optimizer
        self.weights                 = optimizer.weights
        self.biases                  = optimizer.biases
        self.regularization_strength = regularization_strength
        self.num_epochs              = num_epochs
        self.batch_size              = batch_size
        self.costs                   = []
        self.logging_enabled         = logging_enabled

    def run(self):
        for epoch in range(self.num_epochs):
            self.log_epoch(epoch)

            for batch_number in range(self.num_batches()):
                examples_batch = self.batch(batch_number, self.examples)
                labels_batch   = self.batch(batch_number, self.labels)

                self.shuffle_in_unison(examples_batch, labels_batch)

                forward_prop = ForwardProp(self.weights, self.biases, examples_batch)
                self.linear_activation, self.nonlinear_activation = forward_prop.run()

                self.current_cost = self.calculate_cost(forward_prop.network_output, labels_batch)
                self.costs.append(self.current_cost)

                weight_gradients, bias_gradients = self.backward_prop(labels_batch)

                self.optimizer.update_parameters(weight_gradients, bias_gradients)
                self.weights = self.optimizer.weights
                self.biases  = self.optimizer.biases

                self.log_batch(batch_number)

                if self.training_complete(): return self.learned_parameters()

        return self.learned_parameters()

    def learned_parameters(self):
        return {
            'costs':   self.costs,
            'weights': self.weights,
            'biases':  self.biases
        }

    def num_batches(self):
        return int(self.examples.shape[1] / self.batch_size)

    def batch(self, batch_number, array):
        start_index = batch_number * self.batch_size
        end_index   = start_index  + self.batch_size
        return array[:, start_index:end_index]

    def training_complete(self):
        return self.cost_converged() or self.cost_below_threshold()

    def cost_converged(self):
        recent_costs     = self.costs[-5:]
        acceptable_delta = 0.001

        if len(recent_costs) < 5: return False

        for i in range(len(recent_costs) - 1):
            delta = abs(recent_costs[i] - recent_costs[i + 1])
            if delta > acceptable_delta: return False

        print(f"Cost has converged. Recent costs => {recent_costs}")
        return True

    def cost_below_threshold(self):
        return self.current_cost <= 0.05

    def current_cost(self):
        return self.costs[-1]

    def backward_prop(self, labels_batch):
        return BackwardProp(self.weights,
                            self.linear_activation,
                            self.nonlinear_activation,
                            labels_batch,
                            self.regularization_strength).run()

    def calculate_cost(self, network_output, labels_batch):
        return Cost(network_output,
                    labels_batch,
                    self.weights,
                    self.regularization_strength).cross_entropy_loss()

    def log_epoch(self, epoch):
        if self.logging_enabled:
            print(f"\n***Epoch {epoch + 1}***")

    def log_batch(self, batch_number):
        if self.logging_enabled and (batch_number % 500) == 0:
            print(f"Batch {batch_number + 1}, Cost {self.current_cost}")

    # Perform identical in-place shuffles on the *columns* of two arrays
    def shuffle_in_unison(self, array1, array2):
        random_state = np.random.get_state()
        np.random.shuffle(array1.T)

        np.random.set_state(random_state)
        np.random.shuffle(array2.T)
