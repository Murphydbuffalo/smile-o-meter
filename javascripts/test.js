'use strict';

const MockData            = require('./mock_data');
const NeuralNet           = require('./neural_net');
const { weights, biases } = require('./params');
const network             = new NeuralNet(mock.pixels(), weights, biases);

network.predict();
