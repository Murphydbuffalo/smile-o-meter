import imp
import numpy as np
import time
import data.loader
import initialize
import forward_prop
import cost
import backward_prop
import gradient_check
import matplotlib.pyplot as pyplot

Loader        = data.loader.Loader
Initialize    = initialize.Initialize
ForwardProp   = forward_prop.ForwardProp
Cost          = cost.Cost
BackwardProp  = backward_prop.BackwardProp
GradientCheck = gradient_check.GradientCheck

d                    = Loader().load().normalize()
network_architecture = [d.Xtrain_norm.shape[0], 5, 3, 3]
weights, biases      = Initialize(network_architecture).weights_and_biases()
learning_rate        = 0.0000000025
costs                = []
start_time           = time.time()

for i in range(1000):
    Z, A = ForwardProp(weights, biases, d.Xtrain_norm).run()
    c    = Cost(A[-1], d.Ytrain).cross_entropy_loss()
    costs.append(c)
    weight_gradients, bias_gradients = BackwardProp(weights, Z, A, d.Ytrain).run()
    weights                          = weights - (learning_rate * weight_gradients)
    biases                           = biases  - (learning_rate * bias_gradients)
    end_time                         = time.time()
    
print("Total training time in seconds:", end_time - start_time)

pyplot.ylabel('Cost')
pyplot.xlabel('Iteration')
pyplot.plot(costs)
pyplot.show()
