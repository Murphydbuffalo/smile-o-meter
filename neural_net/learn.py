import numpy as np
import os

from lib.optimizers.adam             import Adam
from lib.optimizers.gradient_descent import GradientDescent
from lib.data.data                   import Data
from lib.model                       import Model

data                        = Data().load()
learning_rates              = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
regularization_strengths    = [0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
number_hidden_layers        = [4, 3, 2, 1]
min_number_hidden_nodes     = [7, 14, 21, 28, 35, 42, 49]
hyperparameter_combinations = []

for learning_rate in learning_rates:
    for regularization_strength in regularization_strengths:
        for num_layers in number_hidden_layers:
            for num_nodes in min_number_hidden_nodes:
                hyperparameter_combinations.append({
                    'algorithm':               Adam,
                    'learning_rate':           learning_rate,
                    'regularization_strength': regularization_strength,
                    'number_hidden_layers':    num_layers,
                    'min_number_hidden_nodes': num_nodes
                })

np.random.shuffle(hyperparameter_combinations)

for hyperparameters in hyperparameter_combinations:
    model = Model(data,
                  hyperparameters['algorithm'],
                  hyperparameters['learning_rate'],
                  hyperparameters['regularization_strength'],
                  hyperparameters['number_hidden_layers'],
                  hyperparameters['min_number_hidden_nodes'])

    if model.hash() in " ".join(os.listdir("./output")):
        printf(f"Skipping model {model.hyperparameter_string()} because it's already been trained")
        continue

    print(f"Training model {model.hyperparameter_string()}")

    model.train()
    model.validate()
    model.log_results()
    model.save_parameters()
    model.graph_training_costs()
