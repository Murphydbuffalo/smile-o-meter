'use strict';

const MockData  = require('./mock_data');
const NeuralNet = require('./neural_net');
const mock      = new MockData();
const network   = new NeuralNet(mock.pixels(), mock.weights(), mock.biases());

network.predict();
