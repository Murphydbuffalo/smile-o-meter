'use strict'

// TODO: Test with some mock data for pixels, weights, and biases. Eg:
// let pixels = [];
// while (pixels.length < 100) {
//   pixels.push((Math.floor(Math.random() * 255) + 1));
// }
//
// let biases = [];
// while (biases.length < 10) {
//   biases.push((Math.floor(Math.random() * 5) + 1));
// }
//
// let weights = [];
// ... etc.
// Maybe mock 2 layers of data? inputs with [100x1] dimensions
// weights with [100x10, 10x3] dimensions
// And biases with [10x1, 3x1]
// ?

const NeuralNetwork = class NeuralNetwork {
  constructor(pixels, weights, biases) {
    this.pixels  = pixels;
    this.weights = weights;
    this.biases  = biases;
  }

  predict() {
    let inputs, linear_activation, nonlinear_activation_function, nonlinear_activation;

    inputs = this.pixels;

    this.weights.forEach(function(w, i) {
      linear_activation             = this.dot_product_plus_biases(w, inputs, this.biases[i]);
      nonlinear_activation_function = this.isLastLayer ? this.softmax : this.relu;
      nonlinear_activation          = nonlinear_activation_function(linear_activation);
      inputs                        = nonlinear_activation;
    });

    return this.max_probability(nonlinear_activation);
  }

  // TODO: Make vector/matrix classes to encapsulate the dot product logic
  dot_product_plus_biases(weights, inputs, biases) {
    return this.dot_product(weights, inputs)
               .map((val, i) => val + biases[i]);
  }

  dot_product(weights, inputs) {
    return weights.map(function(column) {
      column.reduce(((sum, weight, i) => sum + (weight * inputs[i])), 0);
    });
  }

  max_probability(probabilities) {
    return Math.max(...probabilities);
  }

  relu(vector) {
    return vector.map((num) => Math.max(num, 0));
  }

  softmax(vector) {
    const exponentials = vector.map((num) => Math.exp(num));
    const sum          = exponentials.reduce(((accumulator, num) => accumulator + num), 0);

    return exponentials.map((exp) => exp / sum);
  }
}

module.exports = NeuralNetwork;
