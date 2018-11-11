import numpy as np
import os

from lib.optimizers.adam       import Adam
from lib.data.data             import Data
from lib.hyperparameter_search import HyperparameterSearch
from lib.model                 import Model

data = Data().load()

for hyperparameters in HyperparameterSearch().hyperparameters():
    model = Model(data,
                  Adam,
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
