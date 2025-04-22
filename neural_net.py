import numpy as np
from autodiff.scalar import Scalar
import random


class Neuron:
    """A single neuron in a neural network.

    Attributes:
        weights: List of weights of the neuron.
        bias: The bias of the neuron.
        activation: The activation function of the neuron.
    """

    def __init__(self, n_inputs, activation, random_state=None):
        """Initialize a neuron.

        Args:
            n_inputs: Number of inputs to the neuron.
            activation: The activation function to use.
            random_state: Random state for reproducibility.
        """
        random.seed(random_state)
        self.activation = activation
        scale = 1 / n_inputs**0.5  # Xavier initialization
        # Initialize weights and bias
        self.weights = [Scalar(random.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.bias = Scalar(0.0)

    def __call__(self, inputs):
        """Compute the output of the neuron.

        Args:
            inputs: List of inputs to the neuron.

        Returns:
            The output of the neuron.
        """
        # Linear transformation: sum(w_i * x_i) + b
        result = Scalar(0.0)
        for i, x in enumerate(inputs):
            result = result + self.weights[i] * x
        result = result + self.bias
        
        # Apply activation function
        return self.activation(result)

    def parameters(self):
        """Get the parameters of the neuron.

        Returns:
            List of parameters (weights and bias).
        """
        return self.weights + [self.bias]


class FeedForwardLayer:
    """A fully-connected feed-forward layer in a neural network.

    Attributes:
        neurons: List of neurons in the layer.
    """

    def __init__(self, n_inputs, n_outputs, activation, random_state=None):
        """Initialize a feed-forward layer.

        Args:
            n_inputs: Number of inputs to the layer.
            n_outputs: Number of outputs of the layer.
            activation: The activation function to use.
            random_state: Random state for reproducibility.
        """
        self.neurons = []
        # Create neurons
        for i in range(n_outputs):
            # Use incremented random_state for each neuron to ensure different initializations
            neuron_random_state = None if random_state is None else random_state + i
            self.neurons.append(
                Neuron(n_inputs, activation, random_state=neuron_random_state)
            )

    def __call__(self, inputs):
        """Compute the outputs of the layer.

        Args:
            inputs: List of inputs to the layer.

        Returns:
            List of outputs of the layer.
        """
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        """Get the parameters of the layer.

        Returns:
            List of parameters of all neurons in the layer.
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MultiLayerPerceptron:
    """A multi-layer perceptron neural network.

    Attributes:
        layers: List of layers in the network.
    """

    def __init__(self, layer_dims, activation, random_state=None):
        """Initialize a multi-layer perceptron.

        Args:
            layer_dims: List of dimensions of the layers.
                E.g., [2, 3, 1] would create a network with
                2 inputs, a hidden layer with 3 neurons, and 1 output.
            activation: The activation function to use for all layers.
            random_state: Random state for reproducibility.
        """
        self.layers = []
        # Create the layers
        for i in range(len(layer_dims) - 1):
            # Use incremented random_state for each layer
            layer_random_state = None if random_state is None else random_state + i * 100
            layer = FeedForwardLayer(
                layer_dims[i],
                layer_dims[i + 1],
                activation,
                random_state=layer_random_state,
            )
            self.layers.append(layer)

    def __call__(self, inputs):
        """Compute the outputs of the network.

        Args:
            inputs: List of inputs to the network.

        Returns:
            List of outputs of the network.
        """
        # Convert inputs to Scalar objects if they are not already
        x = [x if isinstance(x, Scalar) else Scalar(x) for x in inputs]
        
        # Forward pass through each layer
        for layer in self.layers:
            x = layer(x)
        
        return x

    def parameters(self):
        """Get the parameters of the network.

        Returns:
            List of parameters of all layers in the network.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


#please comments are self explanatory, Babs...if any ambiguity, please ask