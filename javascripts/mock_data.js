'use strict'

const MockData = class MockData {
  constructor(network_shape) {
    this.network_shape = network_shape || [2304, 100, 10, 3];
  }

  pixels() {
    let pixels = [];
    while (pixels.length < this.network_shape[0]) {
      pixels.push((Math.floor(Math.random() * 255) + 1));
    }

    return pixels;
  }

  biases() {
    let biases = [];
    for (let i = 1; i < this.network_shape.length; i++) {
      biases.push(this.biasesForLayer(i));
    }

    return biases;
  }

  weights() {
    let weights = [];
    for (let i = 1; i < this.network_shape.length; i++) {
      weights.push(this.weightsForLayer(i));
    }

    return weights;
  }

  weightsForLayer(layer) {
    let weights       = [];
    const num_rows    = this.network_shape[layer - 1];
    const num_columns = this.network_shape[layer];

    for (let i = 0; i < num_columns; i++) {
      let column = [];

      for (let j = 0; j < num_rows; j++) {
        column.push(Math.random() * 0.1);
      }

      weights.push(column);
    }

    return weights;
  }

  biasesForLayer(layer) {
    let biases = [];
    while (biases.length < this.network_shape[layer]) {
      biases.push(Math.random() * 0.1);
    }

    return biases;
  }
}

module.exports = MockData;
