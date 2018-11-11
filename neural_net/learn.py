from lib.optimizers.adam       import Adam
from lib.data.data             import Data
from lib.hyperparameter_search import HyperparameterSearch
from lib.model                 import Model

data = Data().load()

for hyperparameters in HyperparameterSearch().hyperparameters():
    Model(data,
          Adam,
          hyperparameters['learning_rate'],
          hyperparameters['regularization_strength'],
          hyperparameters['number_hidden_layers'],
          hyperparameters['min_number_hidden_nodes']).train()
