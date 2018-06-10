import sys
import numpy as np
import boto3
import json

s3           = boto3.client('s3')
weights_file = s3.download_file('smile-o-meter.rocks', 'learned_weights.npy', '/tmp/learned_weights.npy')
biases_file  = s3.download_file('smile-o-meter.rocks', 'learned_biases.npy',  '/tmp/learned_biases.npy')
weights      = np.load('/tmp/learned_weights.npy')
biases       = np.load('/tmp/learned_biases.npy')

training_set_means_file                    = s3.download_file('smile-o-meter.rocks', 'training_set_means.npy', '/tmp/training_set_means.npy')
standard_deviations_file                   = s3.download_file('smile-o-meter.rocks', 'zero_mean_training_set_standard_deviations.npy',  '/tmp/zero_mean_training_set_standard_deviations.npy')
training_set_means                         = np.load('/tmp/training_set_means.npy')
zero_mean_training_set_standard_deviations = np.load('/tmp/zero_mean_training_set_standard_deviations.npy')

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

# Normalizes input data based on the zero-mean one-standard-deviation
# transformation that was applied to the training set
def normalize(matrix):
    return (matrix - training_set_means) / zero_mean_training_set_standard_deviations

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
