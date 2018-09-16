import numpy as np

from lib.data.data       import Data
from lib.data.formatter  import Formatter
from lib.initialize      import Initialize
from lib.optimize        import Optimize
from lib.optimizers.adam import Adam
from lib.utilities.timer import Timer
from lib.utilities.graph import Graph

unformatted_data     = Data().load()
data                 = Formatter(unformatted_data).run()
network_architecture = [data.num_features, 28, 14, data.num_classes]
weights, biases      = Initialize(network_architecture).weights_and_biases()
timer                = Timer()
optimizer            = Optimize(data.training_examples,
                                data.training_labels,
                                Adam(weights, biases))

timer.time(optimizer.run)
results      = timer.result
time_elapsed = timer.time_elapsed

print(f"Total training was {timer.string()}")

print("Saving learned parameters...")
np.save('./output/learned_weights', results['weights'])
np.save('./output/learned_biases', results['biases'])

print("Saving statistics for normalization of test data...")
np.save('./output/training_set_means', data.training_set_means)
np.save('./output/zero_mean_training_set_standard_deviations', data.zero_mean_training_set_standard_deviations)

Graph(ylabel = "Cost", xlabel = "Iteration", data = results['costs']).render()
