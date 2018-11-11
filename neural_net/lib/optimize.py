import numpy as np

from lib.forward_prop    import ForwardProp
from lib.cost            import Cost
from lib.backward_prop   import BackwardProp

class Optimize:
    def __init__(self, examples, labels, optimizer, regularization_strength, num_epochs = 100, batch_size = 256, logging_enabled = True):
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
            for batch_number in range(self.num_batches()):
                examples_batch = self.batch(batch_number, self.examples)
                labels_batch   = self.batch(batch_number, self.labels)

                self.shuffle_in_unison(examples_batch, labels_batch)

                forward_prop = ForwardProp(self.weights, self.biases, examples_batch)
                self.linear_activation, self.nonlinear_activation = forward_prop.run()

                self.current_cost = self.calculate_cost(forward_prop.network_output, labels_batch)

                weight_gradients, bias_gradients = self.backward_prop(labels_batch)

                self.optimizer.update_parameters(weight_gradients, bias_gradients)
                self.weights = self.optimizer.weights
                self.biases  = self.optimizer.biases

            self.costs.append(self.current_cost)
            self.log_epoch(epoch)

            if self.training_complete():
                return self.learned_parameters()

            if epoch % (self.num_epochs / 4) == 0:
                self.optimizer.learning_rate = self.optimizer.learning_rate * 0.5

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
        return self.cost_below_threshold() or self.cost_not_decreasing()

    def cost_below_threshold(self):
        return self.current_cost <= 0.05

    def cost_not_decreasing(self):
        return len(self.costs) > 4 and self.costs[-1] >= self.costs[-5]

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
            print(f"\nEnd of epoch {epoch + 1} - Cost {round(self.current_cost, 4)}")

    # Perform identical in-place shuffles on the *columns* of two arrays
    def shuffle_in_unison(self, array1, array2):
        random_state = np.random.get_state()
        np.random.shuffle(array1.T)

        np.random.set_state(random_state)
        np.random.shuffle(array2.T)
