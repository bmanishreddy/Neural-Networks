import numpy as np

class NuralNet(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)

        self.sizes = sizes
        print("size  =",self.sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:] ]

        print("bias = ",self.biases)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        print("weights=",self.weights)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            print("The values of a = ",a)
        return a

    # misc files




#print(sigmoid_prime(7))


def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(z))
    return result

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


obj = NuralNet([3,3,2])
print("sigmoid value = ",obj.feedforward(10))
