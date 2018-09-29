import unittest

from lib.initialize import Initialize

class TestInitialize(unittest.TestCase):
    def setUp(self):
        network_architecture = [2304, 250, 150, 7]
        self.weights, self.biases = Initialize(network_architecture).weights_and_biases()

    def test_weights_shape(self):
        self.assertEqual(len(self.weights), 3)
        self.assertEqual(self.weights[0].shape, (250, 2304))
        self.assertEqual(self.weights[1].shape, (150, 250))
        self.assertEqual(self.weights[2].shape, (7, 150))

    def test_weights_magnitude(self):
        self.assertTrue(abs(self.weights[0].mean()) < abs(self.weights[1].mean()))
        self.assertTrue(abs(self.weights[1].mean()) < abs(self.weights[2].mean()))
        self.assertTrue(abs(self.weights[2].mean()) < 0.5)

    def test_biases_shape(self):
        self.assertEqual(len(self.biases), 3)
        self.assertEqual(self.biases[0].shape, (250, 1))
        self.assertEqual(self.biases[1].shape, (150, 1))
        self.assertEqual(self.biases[2].shape, (7, 1))

    def test_biases_magnitude(self):
        self.assertTrue((self.biases[0] == 0).all())
        self.assertTrue((self.biases[1] == 0).all())
        self.assertTrue((self.biases[2] == 0).all())

if __name__ == '__main__':
    unittest.main()
