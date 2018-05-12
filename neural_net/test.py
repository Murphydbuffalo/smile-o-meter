import numpy as np

from lib.data.loader  import Loader
from lib.forward_prop import ForwardProp
from lib.cost         import Cost

d       = Loader().load().normalize()
lambd   = 0.001
X_test  = d.Xtest_norm
Y_test  = d.Ytest

weights = np.load('./output/learned_weights.npy')
biases  = np.load('./output/learned_biases.npy')

Z, A = ForwardProp(weights, biases, X_test).run()
c    = Cost(A[-1], Y_test, weights, lambd).cross_entropy_loss()

print("Average cost:", c)

predictions           = (A[-1] == np.max(A[-1], axis=0)) * 1
m                     = Y_test.shape[1]
correct_predictions   = 0
incorrect_predictions = 0

for i in range(m):
    if (predictions[:,i] == Y_test[:,i]).all():
        correct_predictions   += 1
    else:
        incorrect_predictions += 1

total_predictions = correct_predictions + incorrect_predictions

print("m:",                     m)
print("total_predictions:",     total_predictions)
print("correct_predictions:",   correct_predictions)
print("incorrect_predictions:", incorrect_predictions)
print("percent correct:",       correct_predictions / total_predictions)
