import numpy             as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot

from time     import time
from datetime import datetime, timedelta
from sys      import argv

from lib.data.sources.fer_csv import FER_CSV
from lib.data.formatter       import Formatter
from lib.initialize           import Initialize
from lib.forward_prop         import ForwardProp
from lib.cost                 import Cost
from lib.backward_prop        import BackwardProp
from lib.optimize             import Adam
from lib.gradient_check       import GradientCheck

pixels_csv = FER_CSV().load_data()
data                 = Formatter(pixels_csv).run()
network_architecture = [data.num_features, 250, 125, data.num_classes]
weights, biases      = Initialize(network_architecture).weights_and_biases()
costs                = []
lambd                = 0.0001
start_time           = time()

momentum_weight_average = np.zeros(weights.shape)
momentum_bias_average   = np.zeros(biases.shape)
rms_prop_weight_average = np.zeros(weights.shape)
rms_prop_bias_average   = np.zeros(biases.shape)

for i in range(1200):
    Z, A = ForwardProp(weights, biases, data.Xtrain_norm).run()
    c    = Cost(A[-1], data.Ytrain, weights, lambd).cross_entropy_loss()
    costs.append(c)

    weight_gradients, bias_gradients = BackwardProp(weights, Z, A, data.Ytrain, lambd).run()

    optimizer = Adam(
        weights,
        biases,
        weight_gradients,
        bias_gradients,
        momentum_weight_average,
        momentum_bias_average,
        rms_prop_weight_average,
        rms_prop_bias_average
    )

    [
        updated_weights,
        updated_biases,
        updated_momentum_weight_average,
        updated_momentum_bias_average,
        updated_rms_prop_weight_average,
        updated_rms_prop_bias_average
    ] = optimizer.updated_parameters()

    if (i % 10) == 0:
        print("Iteration #", i)
        print("Cost is", c)

        if len(argv) > 1 and argv[1] == '--check-gradients' and i % 100 == 0:
            check = GradientCheck(weights, biases, weight_gradients, data.Xtrain_norm, data.Ytrain)
            print("Are the analytic gradients about the same as the numeric gradients?", check.run())

    weights                 = updated_weights
    biases                  = updated_biases
    momentum_weight_average = updated_momentum_weight_average
    momentum_bias_average   = updated_momentum_bias_average
    rms_prop_weight_average = updated_rms_prop_weight_average
    rms_prop_bias_average   = updated_rms_prop_bias_average

end_time     = time()
seconds      = timedelta(seconds=int(end_time - start_time))
time_elapsed = datetime(1,1,1) + seconds
print(f"Total training was {time_elapsed.day - 1}:{time_elapsed.hour}:{time_elapsed.minute}:{time_elapsed.second}")

print("Saving learned parameters...")
np.save('./output/learned_weights', weights)
np.save('./output/learned_biases', biases)

print("Saving statistics for normalization...")
np.save('./output/training_set_means', data.training_set_means)
np.save('./output/zero_mean_training_set_standard_deviations', data.zero_mean_training_set_standard_deviations)

pyplot.ylabel('Cost')
pyplot.xlabel('Iteration')
pyplot.plot(costs)
pyplot.show()
