import numpy as np
import os

from lib.utilities.confusion_matrix import ConfusionMatrix
from lib.utilities.graph            import Graph
from lib.initialize                 import Initialize
from lib.optimize                   import Optimize
from lib.predict                    import Predict
from lib.utilities.timer            import Timer

class Model:
    def __init__(self, data, optimization_algorithm, learning_rate, regularization_strength, num_layers, min_number_hidden_nodes):
        self.data                    = data
        self.optimization_algorithm  = optimization_algorithm
        self.learning_rate           = learning_rate
        self.regularization_strength = regularization_strength
        self.num_hidden_layers       = num_layers
        self.min_number_hidden_nodes = min_number_hidden_nodes
        self.timer                   = Timer()
        self.hyperparameters         = {
            'Algorithm':               optimization_algorithm.__name__,
            'Learning Rate':           learning_rate,
            'Regularization Strength': regularization_strength,
            'Num Hidden Layers':       num_layers,
            'Min Number Hidden Nodes': min_number_hidden_nodes
        }

    def train(self):
        if self.hash() in " ".join(os.listdir("./output")):
            print(f"Model {self.hash()} has already been trained. You can find its learned parameters and data about its effectiveness in {self.dirname()}.")
            return

        print(f"Training model {self.hash()}")

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
        self.save_output()

    def initialize_parameters(self):
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

        validation_results     = predictor.run()
        self.accuracy          = round(validation_results['accuracy'], 4)
        self.cost              = round(validation_results['cost'], 4)
        self.actual_classes    = predictor.actual_classes
        self.predicted_classes = predictor.predicted_classes()

    def save_output(self):
        os.mkdir(self.dirname())

        np.save(self.dirname() + "/weights", self.learned_weights)
        np.save(self.dirname() + "/biases",  self.learned_biases)

        with open(self.dirname() + "/results.txt", "a") as results_file:
            results_file.write(self.results_string())
            results_file.close()

        self.graph_costs()

        print(f"Results saved to {self.dirname()}\n")

    def results_string(self):
        string = f"Network architecture:\n{self.network_architecture()}\n"
        string = string + "Hyperparameters:\n"

        for key, value in self.hyperparameters.items():
            string = string + f"{key} => {value}\n"

        string = string + f"\nTotal training time => {self.timer.string()}\n"
        string = string + f"\nAverage validation cost => {self.cost}\n"
        string = string + f"Validation accuracy => {self.accuracy}\n"
        string = string + f"\nConfusion Matrix:\n{self.confusion_matrix()}\n"

        return string

    def dirname(self):
        return ("./output/accuracy-{0}-{1}").format(self.accuracy, self.hash())

    def hash(self):
        hyperparameter_string = '-'.join([f"{key}-{value}" for key, value in self.hyperparameters.items()])
        return str(hash(hyperparameter_string))

    def confusion_matrix(self):
        return ConfusionMatrix(self.actual_classes, self.predicted_classes).string()

    def graph_costs(self):
        graph = Graph(ylabel = "Cost", xlabel = "Iteration", data = self.training_costs)
        graph.save(f"{self.dirname()}/costs.png")
