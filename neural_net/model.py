import numpy as np
import copy  as copy
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
Z, A            = ForwardProp(weights, biases, d.Xtrain_norm).run()

# To calculate the cost of the network we take only the softmax predictions for
# the class corresponding to the label, y, for each example.
#
# When using the network to make predictions we'll keep the predictions for all
# classes and choose the one with the highest probability.
predictions                      = np.choose(d.Ytrain, A[-1])
c                                = Cost(predictions, d.Ytrain).cross_entropy_loss()
weight_gradients, bias_gradients = BackwardProp(Z, A, d.Ytrain).run()
