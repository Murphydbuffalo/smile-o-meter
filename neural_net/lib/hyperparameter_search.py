import numpy as np

learning_rates              = [1e-01, 1e-02, 1e-03, 1e-04, 1e-5]
regularization_strengths    = [1e-02, 1e-03, 1e-04, 1e-05, 1e-6]
number_hidden_layers        = [1, 2, 3]
min_number_hidden_nodes     = [10, 100, 1000]
hyperparameter_combinations = []

class HyperparameterSearch:
    def hyperparameters(self):
        for learning_rate in learning_rates:
            for regularization_strength in regularization_strengths:
                for num_layers in number_hidden_layers:
                    for num_nodes in min_number_hidden_nodes:
                        hyperparameter_combinations.append({
                            'learning_rate':           learning_rate,
                            'regularization_strength': regularization_strength,
                            'number_hidden_layers':    num_layers,
                            'min_number_hidden_nodes': num_nodes
                        })

        np.random.shuffle(hyperparameter_combinations)

        return hyperparameter_combinations
