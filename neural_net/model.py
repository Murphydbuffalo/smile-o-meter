import numpy as np
from data.loader  import Loader
from initialize   import Initialize
from forward_prop import ForwardProp
from cost         import Cost

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
predictions     = ForwardProp(weights, biases, d.Xtrain_norm).run()
c               = Cost(predictions, d.Ytrain).cross_entropy_loss()

# NEXT STEPS:
# Write out the normalized data to CSVs and write code to import those CSVs into
# NumPy arrays. This will prevent you from having to generate this data every time
# you want to run your model.
#
# Replace sigmoid with softmax.
#
# Cost function for softmax.
#
# Backprop (probably will need to go back and update forward prop to cache certain values)
# for use in backprop.
#
# print(len(weights))
# for matrix in weights:
#     print(matrix)
#
# print(len(biases))
# for matrix in biases:
#     print(matrix)
# print('Xtrain is a', type(d.Xtrain))
# print('Xtrain.shape is', d.Xtrain.shape)
# print('Xtrain is', d.Xtrain)
#
# print('Xtrain_norm is', d.Xtrain_norm)
# print('Xtrain_norm mean is', np.mean(d.Xtrain_norm, 0))
# print('Xtrain_norm standard deviation is', np.std(d.Xtrain_norm, 0))
#
# print('Ytrain is a', type(d.Ytrain))
# print('Ytrain.shape is', d.Ytrain.shape)
# print('Ytrain is', d.Ytrain)
#
# print('Xdev is a', type(d.Xdev))
# print('Xdev.shape is', d.Xdev.shape)
# print('Xdev is', d.Xdev)
#
# print('Xdev_norm is', d.Xdev_norm)
# print('Xdev_norm mean is', np.mean(d.Xdev_norm, 0))
# print('Xdev_norm standard deviation is', np.std(d.Xdev_norm, 0))
#
# print('Ydev is a', type(d.Ydev))
# print('Ydev.shape is', d.Ydev.shape)
# print('Ydev is', d.Ydev)
#
# print('Xtest is a', type(d.Xtest))
# print('Xtest.shape is', d.Xtest.shape)
# print('Xtest is', d.Xtest)
#
# print('Xtest_norm is', d.Xtest_norm)
# print('Xtest_norm mean is', np.mean(d.Xtest_norm, 0))
# print('Xtest_norm standard deviation is', np.std(d.Xtest_norm, 0))
#
# print('Ytest is a', type(d.Ytest))
# print('Ytest.shape is', d.Ytest.shape)
# print('Ytest is', d.Ytest)
