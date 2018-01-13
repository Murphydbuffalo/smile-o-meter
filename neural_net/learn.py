import numpy as np
from data.loader   import Loader
from initialize    import Initialize
from forward_prop  import ForwardProp
from cost          import Cost
from backward_prop import BackwardProp

d = Loader()
d.load()
d.normalize()

network_architecture = [
    d.Xtrain_norm.shape[0],
    5,
    3,
    3
]

weights, biases = Initialize(network_architecture).weights_and_biases()

for i in range(10000):
    Z, A                             = ForwardProp(weights, biases, d.Xtrain_norm).run()
    c                                = Cost(A[-1], d.Ytrain).cross_entropy_loss()
    weight_gradients, bias_gradients = BackwardProp(weights, Z, A, d.Ytrain).run()
    weights                          = weights - (0.000001 * weight_gradients)
    biases                           = biases  - (0.000001 * bias_gradients)
    if (i % 10) == 0:
        print("Iteration #", i + 1)
        print("Cost:", c)
