'use strict'

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
