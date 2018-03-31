'use strict';

const fs     = require('fs');
const format = 'utf8';

try {
  // TODO: Host the JSON of the learned parameters in an S3 bucket and fetch() them
  const weights = JSON.parse(fs.readFileSync('learned_params/learned_weights.json', format);
  const biases  = JSON.parse(fs.readFileSync('learned_params/learned_biases.json',  format);
} catch (e) {
  console.error('Error loading parameters:', e.message);
}

module.exports = { weights, biases };
