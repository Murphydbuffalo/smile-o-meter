import numpy             as np

from lib.data.loader    import Loader
from lib.initialize     import Initialize
from lib.forward_prop   import ForwardProp
from lib.cost           import Cost
from lib.backward_prop  import BackwardProp
from lib.optimize       import Adam
from lib.gradient_check import GradientCheck

d                    = Loader().load().normalize()
network_architecture = [d.Xtrain_norm.shape[0], 100, 10, 3]
weights, biases      = Initialize(network_architecture).weights_and_biases()
costs                = []
lambd                = 0.001

momentum_weight_average = np.zeros(weights.shape)
momentum_bias_average   = np.zeros(biases.shape)
rms_prop_weight_average = np.zeros(weights.shape)
rms_prop_bias_average   = np.zeros(biases.shape)
