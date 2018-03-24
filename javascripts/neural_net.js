'use strict'

const NeuralNetwork = class NeuralNetwork {
  constructor(pixels, weights, biases) {
    this.pixels  = pixels;
    this.weights = weights;
    this.biases  = biases;
  }

  predict() {
    let input, linear_activation, nonlinear_activation_function, nonlinear_activation;

    inputs = this.pixels;

    this.weights.forEach(function(w, i) {
      linear_activation             = this.dot_product_plus_biases(w, inputs, this.biases[i]);
      nonlinear_activation_function = this.isLastLayer ? this.relu : this.softmax;
      nonlinear_activation          = nonlinear_activation_function(linear_activation);
      inputs                        = nonlinear_activation;
    });

    return this.vector_to_int(nonlinear_activation);
  }

  // will be working with m = 1 in this case
  // so first layer should be:
  // weights   = [[...].length == 100].length == 2304
  // input     = [...].length == 2304
  // biases    = [...].length == 100
  // W*X       = [...].length == 100
  // A = W*X+B = [...].length == 100

  dot_product_plus_biases(weights, inputs, biases) {
    return this.dot_product(weights, inputs)
               .map((val, i) => val + biases[i]);
  }

  dot_product(weights, inputs) {
    return weights.map(function(column) {
      column.reduce(((sum, weight, i) => sum + (weight * inputs[i])), 0);
    });
  }

  vector_to_int(vector) {
    return vector.indexOf(1);
  }

  relu(vector) {
    return vector.map((num) => num > 0 ? num : 0);
  }

  softmax(vector) {
    const exponentials = vector.map((num) => Math.exp(num));
    const sum          = exponentials.reduce(((accumulator, num) => accumulator + num), 0);

    return exponentials.map((exp) => exp / sum);
  }
}