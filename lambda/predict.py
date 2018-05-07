import numpy as np
import boto3

s3           = boto3.client('s3')
weights_file = s3.download_file('smile-o-meter.rocks', 'learned_weights.npy', '/tmp/learned_weights.npy')
biases_file  = s3.download_file('smile-o-meter.rocks', 'learned_biases.npy',  '/tmp/learned_biases.npy')
weights      = np.load('/tmp/learned_weights.npy')
biases       = np.load('/tmp/learned_biases.npy')

def relu(Z):
    return np.maximum(Z, 0)

def softmax(v):
    exponentials = np.exp(v - np.max(v, axis = 0))
    return exponentials / np.sum(exponentials)

def softmax_activation(Z):
    return np.atleast_2d(np.apply_along_axis(softmax, 0, Z))

def forward_prop(A):
    for i in range(len(weights)):
        Z = np.dot(weights[i], A) + biases[i]
        A = relu(Z)

    return softmax_activation(Z)

def predict(event, context):
    pixels             = np.array([event['pixels']]).T
    softmax_activation = forward_prop(pixels)
    prediction         = np.argmax(softmax_activation)

    return { "statusCode": 200, "body": str(prediction) }
