'use strict';

const fs     = require('fs');
const format = 'utf8';

let weights, biases;

try {
  // TODO: Host the JSON of the learned parameters in an S3 bucket and fetch() them
  weights = JSON.parse(fs.readFileSync('../neural_net/weights.json', format));
  biases  = JSON.parse(fs.readFileSync('../neural_net/biases.json',  format));
} catch (e) {
  console.error('Error loading parameters:', e.message);
}

module.exports = { weights, biases };
