import numpy as np
import os

from lib.utilities.graph import Graph
from lib.initialize      import Initialize
from lib.optimize        import Optimize
from lib.predict         import Predict
from lib.utilities.timer import Timer

class Model:
    def __init__(self, data, optimization_algorithm, learning_rate, regularization_strength, num_layers, min_number_hidden_nodes):
        self.data                    = data
        self.optimization_algorithm  = optimization_algorithm
        self.learning_rate           = learning_rate
        self.regularization_strength = regularization_strength
        self.num_hidden_layers       = num_layers
        self.min_number_hidden_nodes = min_number_hidden_nodes
        self.timer                   = Timer()

    def train(self):
        if self.hyperparameter_string() in " ".join(os.listdir("./output")):
            print(f"Model {self.hyperparameter_string()} has already been trained. You can find its learned parameters in ./output")
            return

        print(f"Training model {self.hyperparameter_string()}")

        weights, biases = self.initialize_parameters()
        algorithm       = self.optimization_algorithm(self.learning_rate, weights, biases)
        optimizer       = Optimize(self.data.training_examples,
                                   self.data.training_labels,
                                   algorithm,
                                   self.regularization_strength)

        training_results = self.timer.time(optimizer.run)

        self.training_costs  = training_results['costs']
        self.learned_weights = training_results['weights']
        self.learned_biases  = training_results['biases']

        self.validate()
        self.log_results()
        self.save_parameters()

    def initialize_parameters(self):
        print("Network architecture:", self.network_architecture())
        return Initialize(self.network_architecture()).weights_and_biases()

    def network_architecture(self):
        num_features = self.data.training_examples.shape[0]
        num_classes  = self.data.training_labels.shape[0]
        architecture = [num_features]

        for hidden_layer_index in range(self.num_hidden_layers):
            architecture.append(self.min_number_hidden_nodes * (self.num_hidden_layers - hidden_layer_index))

        architecture.append(num_classes)

        return architecture

    def validate(self):
        predictor = Predict(self.data.validation_examples,
                            self.data.validation_labels,
                            self.learned_weights,
                            self.learned_biases)

        validation_results = predictor.run()
        self.accuracy      = validation_results['accuracy']
        self.cost          = validation_results['cost']

        return validation_results

    def log_results(self):
        print("\nTRAINING INFO")
        print(f"Total training time: {self.timer.string()}")
        print(f"Algorithm: {self.optimization_algorithm.__name__}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Regularization strength: {self.regularization_strength}")
        print(f"Number of hidden layers: {self.num_hidden_layers}")
        print(f"Number of hidden nodes in smallest layer: {self.min_number_hidden_nodes}")

        print("\nVALIDATION RESULTS")
        print("Accuracy:",     self.accuracy)
        print("Average cost:", self.cost)

    def save_parameters(self):
        np.save(self.weights_filename(), self.learned_weights)
        np.save(self.biases_filename(),  self.learned_biases)

    def weights_filename(self):
        return ("./output/accuracy-{0}-{1}-weights").format(self.accuracy, self.hyperparameter_string())

    def biases_filename(self):
        return ("./output/accuracy-{0}-{1}-biases").format(self.accuracy, self.hyperparameter_string())

    def hyperparameter_string(self):
        hyperparameters = ['algorithm',
                           self.algorithm_name(),
                           'learning-rate',
                           self.learning_rate,
                           'regularization-strength',
                           self.regularization_strength,
                           'num-hidden-layers',
                           self.num_hidden_layers,
                           'min-number-hidden-nodes',
                           self.min_number_hidden_nodes]

        return '-'.join([str(hp) for hp in hyperparameters])

    def algorithm_name(self):
        return self.optimization_algorithm.__name__

    def graph_training_costs(self):
        Graph(ylabel = "Cost", xlabel = "Iteration", data = self.training_costs).render()
