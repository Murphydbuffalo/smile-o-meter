import sys
import numpy as np
import boto3
import json

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

# Normalizes input data to have mean 0 and variance 1
def normalize(matrix):
    means                                 = np.mean(matrix, 0)
    mean_zero_data                        = matrix - means
    standard_deviations                   = np.std(mean_zero_data, 0)

    return mean_zero_data / standard_deviations

def predict(event, context):
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
    }

    try:
        pixels             = json.loads(event['body'])['pixels']
        array              = np.array([pixels]).T
        normalized_array   = normalize(array)
        softmax_activation = forward_prop(normalized_array)
        prediction         = np.argmax(softmax_activation)

        return {
            "statusCode": 200,
            "body": str(prediction),
            "headers": headers
        }
    except:
        return {
            "statusCode": 500,
            "body": str(sys.exc_info()),
            "headers": headers
        }
