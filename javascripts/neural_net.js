'use strict'

const NeuralNetwork = class NeuralNetwork {
  constructor(pixels, weights, biases) {
    this.pixels  = pixels;
    this.weights = weights;
    this.biases  = biases;
  }

  predict() {
    let inputs, linearActivation, nonlinearActivationFunction, nonlinearActivation;

    inputs = this.pixels;

    this.weights.forEach((w, i) => {
      linearActivation            = this.dotProductPlusBiases(w, inputs, this.biases[i]);
      nonlinearActivationFunction = this.isLastLayer ? this.softmax : this.relu;
      nonlinearActivation         = nonlinearActivationFunction(linearActivation);
      inputs                      = nonlinearActivation;
    });

    return this.max_probability(nonlinear_activation);
  }

  // TODO: Make vector/matrix classes to encapsulate the dot product logic
  dotProductPlusBiases(weights, inputs, biases) {
    return this.dotProduct(weights, inputs)
               .map((val, i) => val + biases[i]);
  }

  dotProduct(weights, inputs) {
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
