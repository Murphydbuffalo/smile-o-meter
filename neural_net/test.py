import numpy as np

from lib.data.sources.fer_csv import FER_CSV
from lib.data.formatter       import Formatter
from lib.predict              import Predict

pixels_csv = FER_CSV().load_data()
data       = Formatter(pixels_csv).run()
weights    = np.load('./output/learned_weights.npy')
biases     = np.load('./output/learned_biases.npy')
predictor  = Predict(data.test_examples, data.test_labels, weights, biases)

predictor.run()

print("Average cost:",                  predictor.cost)
print("Number of examples:",            predictor.num_examples)
print("Number of correct predictions:", predictor.num_correct)
print("Percent correct:",               predictor.percent_correct)
