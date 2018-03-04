import numpy             as np
import matplotlib.pyplot as pyplot

from time     import time
from datetime import datetime, timedelta
from sys      import argv

from data.loader    import Loader
from initialize     import Initialize
from forward_prop   import ForwardProp
from cost           import Cost
from backward_prop  import BackwardProp
from optimize       import Adam
from gradient_check import GradientCheck

d                    = Loader().load().normalize()
network_architecture = [d.Xtrain_norm.shape[0], 5, 3, 3]
weights, biases      = Initialize(network_architecture).weights_and_biases()
costs                = []
start_time           = time()

momentum_weight_average = np.zeros(weights.shape)
momentum_bias_average   = np.zeros(biases.shape)
rms_prop_weight_average = np.zeros(weights.shape)
rms_prop_bias_average   = np.zeros(biases.shape)

for i in range(10_000):
    Z, A = ForwardProp(weights, biases, d.Xtrain_norm).run()
    c    = Cost(A[-1], d.Ytrain).cross_entropy_loss()

    if (len(costs) > 0 and costs[-1] < c):
        print("~~~~~ Cost has increased ~~~~~")

    costs.append(c)

    weight_gradients, bias_gradients = BackwardProp(weights, Z, A, d.Ytrain).run()

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

        if len(argv) > 1 and argv[1] == '--check-gradients':
            check = GradientCheck(weights, biases, weight_gradients, d.Xtrain_norm, d.Ytrain)
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
np.save('learned_weights', weights)
np.save('learned_biases', biases)

pyplot.ylabel('Cost')
pyplot.xlabel('Iteration')
pyplot.plot(costs)
pyplot.show()
