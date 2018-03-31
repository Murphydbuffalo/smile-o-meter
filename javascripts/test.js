'use strict';

const MockData            = require('./mock_data');
const mock                = new MockData();
const { weights, biases } = require('./params');

const NeuralNet           = require('./neural_net');
const network             = new NeuralNet(mock.pixels(), weights, biases);

network.predict();
