import imp
import numpy as np
import data.loader
import initialize
import forward_prop
import cost
import backward_prop
import gradient_check

Loader        = data.loader.Loader
Initialize    = initialize.Initialize
ForwardProp   = forward_prop.ForwardProp
Cost          = cost.Cost
BackwardProp  = backward_prop.BackwardProp
GradientCheck = gradient_check.GradientCheck

d                    = Loader().load().normalize()
network_architecture = [d.Xtrain_norm.shape[0], 5, 3, 3]
weights, biases      = Initialize(network_architecture).weights_and_biases()

for i in range(10_000):
    Z, A                             = ForwardProp(weights, biases, d.Xtrain_norm).run()
    c                                = Cost(A[-1], d.Ytrain).cross_entropy_loss()
    weight_gradients, bias_gradients = BackwardProp(weights, Z, A, d.Ytrain).run()
    weights                          = weights - (0.0000000025 * weight_gradients)
    biases                           = biases  - (0.0000000025 * bias_gradients)
    if (i % 10) == 0:
        print("Iteration #", i + 1)
        print("Cost:", c)
        # if (i % 500) == 0:
        #     print("Gradients OK?", GradientCheck(weights, biases, weight_gradients, d.Xtrain_norm, d.Ytrain).close_enough())
