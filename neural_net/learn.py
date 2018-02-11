import numpy as np
import data.loader
import initialize
import forward_prop
import cost
import backward_prop
import optimize
import matplotlib.pyplot as pyplot
from time import time
from datetime import datetime, timedelta
import gradient_check

Loader          = data.loader.Loader
Initialize      = initialize.Initialize
ForwardProp     = forward_prop.ForwardProp
Cost            = cost.Cost
BackwardProp    = backward_prop.BackwardProp
GradientDescent = optimize.GradientDescent

d                    = Loader().load().normalize()
network_architecture = [d.Xtrain_norm.shape[0], 5, 3, 3]
weights, biases      = Initialize(network_architecture).weights_and_biases()
costs                = []
start_time           = time()

for i in range(1000):
    Z, A = ForwardProp(weights, biases, d.Xtrain_norm).run()
    c    = Cost(A[-1], d.Ytrain).cross_entropy_loss()
    costs.append(c)

    weight_gradients, bias_gradients = BackwardProp(weights, Z, A, d.Ytrain).run()

    if (i % 100) == 0:
        print("Iteration #", i)
        print("Cost is", c)
        check = gradient_check.GradientCheck(weights, biases, weight_gradients, d.Xtrain_norm, d.Ytrain)
        print("Are the analytic gradients about the same as the numeric gradients?", check.run())

    updated_weights, updated_biases  = GradientDescent(weights, biases, weight_gradients, bias_gradients).updated_parameters()
    weights                          = updated_weights
    biases                           = updated_biases

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
