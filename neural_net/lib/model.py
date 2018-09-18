from lib.optimizers.adam import Adam
from lib.utilities.graph import Graph
from lib.initialize      import Initialize
from lib.optimize        import Optimize
from lib.predict         import Predict
from lib.utilities.timer import Timer

class Model:
    def __init__(self, data, learning_rate, regularization_strength, num_layers, min_number_hidden_nodes):
        self.data                    = data
        self.learning_rate           = learning_rate
        self.regularization_strength = regularization_strength
        self.num_hidden_layers       = num_layers
        self.min_number_hidden_nodes = min_number_hidden_nodes
        self.timer                   = Timer()

    def validate(self):
        assert hasattr(self, 'learned_weights'), 'You must train the model before it can be validated!'

        predictor = Predict(self.data.validation_examples,
                            self.data.validation_labels,
                            self.learned_weights,
                            self.learned_biases)

        validation_results = predictor.run()
        self.accuracy = validation_results['accuracy']
        self.cost     = validation_results['cost']

        return validation_results

    def train(self):
        optimizer = Optimize(self.data.training_examples,
                             self.data.training_labels,
                             self.optimization_algorithm(),
                             self.regularization_strength)

        training_results = self.timer.time(optimizer.run)

        self.training_costs  = training_results['costs']
        self.learned_weights = training_results['weights']
        self.learned_biases  = training_results['biases']

        return training_results

    def optimization_algorithm(self):
        weights, biases = self.initialize_parameters()
        return Adam(self.learning_rate, weights, biases)

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

    def log_results(self):
        print("\nTRAINING INFO")
        print(f"Total training time: {self.timer.string()}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Regularization strength: {self.regularization_strength}")
        print(f"Number of hidden layers: {self.num_hidden_layers}")
        print(f"Number of hidden nodes in smallest layer: {self.min_number_hidden_nodes}")

        print("\nVALIDATION RESULTS")
        print("Accuracy:",     self.accuracy)
        print("Average cost:", self.cost)


    def save_parameters(self):
        filename_prefix = (
            "./output/accuracy-{0}-learning_rate_{1}_regularization_{2}_num_layers_{3}_num_nodes_{4}"
        ).format(self.validation_results['accuracy'],
                 self.learning_rate,
                 self.regularization_strength,
                 self.num_hidden_layers,
                 self.min_number_hidden_nodes)

        np.save(filename_prefix + '_weights', results['weights'])
        np.save(filename_prefix + '_biases',  results['biases'])

    def graph_training_costs(self):
        Graph(ylabel = "Cost", xlabel = "Iteration", data = self.training_costs).render()
