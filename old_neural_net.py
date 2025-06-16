import numpy as np
from autodiff.scalar import Scalar
import random

class Neuron:
    def __init__(self, n_inputs, activation, random_state=None):
        random.seed(random_state)
        self.activation = activation
        scale = 1 / n_inputs**0.5
        self.weights = [Scalar(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Scalar(0.0)

    def __call__(self, inputs):
        result = Scalar(0.0)
        for i, x in enumerate(inputs):
            result += self.weights[i] * x
        return self.activation(result + self.bias)

    def parameters(self):
        return self.weights + [self.bias]

class FeedForwardLayer:
    def __init__(self, n_inputs, n_outputs, activation, random_state=None):
        self.neurons = []
        for i in range(n_outputs):
            seed = None if random_state is None else random_state + i
            self.neurons.append(Neuron(n_inputs, activation, random_state=seed))

    def __call__(self, inputs):
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MultiLayerPerceptron:
    def __init__(self, layer_dims, activation, random_state=None):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            seed = None if random_state is None else random_state + i * 100
            self.layers.append(FeedForwardLayer(layer_dims[i], layer_dims[i+1], activation, seed))

    def __call__(self, inputs):
        x = [Scalar(x) if not isinstance(x, Scalar) else x for x in inputs]
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
