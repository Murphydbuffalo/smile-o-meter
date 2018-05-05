import numpy as np
import boto3

s3           = boto3.client('s3')
weights_file = s3.download_file('smile-o-meter.rocks', 'learned_weights.npy', 'learned_weights.npy')
biases_file  = s3.download_file('smile-o-meter.rocks', 'learned_biases.npy',  'learned_biases.npy')
weights      = np.load('learned_weights.npy')
biases       = np.load('learned_biases.npy')

def predict(event, context):
    softmax_activation = forward_prop(event['pixels'])
    return np.argmax(softmax_activation)

def forward_prop(A):
    for i in range(len(weights)):
        Z = np.dot(weights[i], A) + biases[i]
        A = relu(Z)

    return softmax_activation(Z)

def relu(Z):
    # NOTE: np.maximum is NOT the same as np.max, which find the maximum value
    # in the matrix (or in each row/column of the matrix).
    # np.maximum maps the given array element-wise, returning the max of each
    # element and the provided second arg (0 in this case)
    return np.maximum(Z, 0)

def softmax(v):
    exponentials = np.exp(v - np.max(v, axis = 0))
    return exponentials / np.sum(exponentials)

def softmax_activation(Z):
    return np.atleast_2d(np.apply_along_axis(softmax, 0, Z))
