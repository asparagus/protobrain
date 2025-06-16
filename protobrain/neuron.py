"""Module dealing with Neurons."""

import logging

from protobrain import computation
from protobrain import learning
from protobrain import synapses


log = logging.getLogger(__name__)


class Neurons:
    """Class representing neurons."""

    MAIN_INPUT = "main"

    def __init__(self, shape, computation=None, learning=None):
        """Initialize the neurons.

        Args:
            units: Either a number of internal units or a list of layers
            computation: Optional - the computation function to use to obtain the outputs
            learning: Optional - the function to use for learning
        """
        if isinstance(shape, int):
            shape = (shape,)

        self.inputs = {self.MAIN_INPUT: synapses.Input(self.MAIN_INPUT, shape=shape)}
        self.output = synapses.Output(shape=shape)
        self.computation = computation
        self.learning = learning

    def compute(self):
        """Compute the output of this neuron."""
        if not self.computation:
            raise ValueError(f"Computation function not set for {self}!")
        self.output.values = self.computation(**self.inputs)
        return self.values

    def learn(self):
        """Adjust the synapses to learn."""
        if not self.computation:
            raise ValueError(f"Learning function not set for {self}!")
        if self.learning:
            self.learning(self)

    @property
    def input(self):
        """Get the main input."""
        return self.get(Neurons.MAIN_INPUT)

    @input.setter
    def input(self, output):
        """Connect the main input to the given output."""
        self.set(Neurons.MAIN_INPUT, output)

    def get(self, name):
        """Get an input to these neurons by name.

        Args:
            name: The name of the input to retrieve

        Returns:
            The input that goes by that name.
        """
        if name not in self.inputs:
            raise IndexError("{0} not set as an input for {1}".format(name, repr(self)))
        return self.inputs[name]

    def set(self, name, output, synapse_function=None):
        """Connect an input to the given output.

        Args:
            name: The name of the input
            output: The output to connect this input to
            synapse_function: Optional - the function to use to generate
                the synapses
        """
        self.inputs[name] = synapses.Input(name, shape=self.output.shape)
        self.inputs[name].connect(output, synapse_function)

    @property
    def values(self):
        """The values at the output of these neurons."""
        return self.output.values

    @property
    def shape(self):
        """The shape of the output of these neurons."""
        return self.output.shape


class LayeredNeurons(Neurons):
    """Class representing groups of neurons."""

    def __init__(self, layers):
        self.layers = layers
        self.inputs = self.layers[0].inputs
        self.output = self.layers[-1].output

        if self.inputs[self.MAIN_INPUT].connected:
            log.warning("Creating layers with pre-connected input")

    def learn(self):
        """Adjust the synapses to learn."""
        for layer in self.layers:
            layer.learn()

    def compute(self):
        """Compute the output of these layers."""
        for layer in self.layers:
            layer.compute()
        return self.values

    @property
    def computation(self):
        return [layer.computation for layer in self.layers]

    @computation.setter
    def computation(self, function):
        if function is None or isinstance(function, computation.Computation):
            function = [function] * len(self.layers)

        for i, layer in enumerate(self.layers):
            layer.computation = function[i]

    @property
    def learning(self):
        return [layer.learning for layer in self.layers]

    @learning.setter
    def learning(self, function):
        if function is None or isinstance(function, learning.Learning):
            function = [function] * len(self.layers)

        for i, layer in enumerate(self.layers):
            layer.learning = function[i]


def FeedForward(layers, input_name="main", synapse_function=None):
    """Connect all the layers in a feed forward fashion.

    Each layer's output will be connected to the next layer's input
    matching the given name.

    Args:
        input_name: The input to which to connect
        synapse_function: Optional - the function to use to generate
            the synapses

    Returns:
        A Neurons object containing the layers
    """
    for i, layer in enumerate(layers[:-1]):
        layers[i + 1].set(input_name, layer)

    return LayeredNeurons(layers)


def FeedBackward(layers, input_name=None, synapse_function=None):
    """Connect all the layers in a feed backward fashion.

    Each layer's output will be connected to the previous layer's input
    matching the given name.

    Args:
        input_name: The input to which to connect
        synapse_function: Optional - the function to use to generate
            the synapses
    Returns:
        A Neurons object containing the layers
    """
    for i, layer in enumerate(layers[:-1]):
        layer.set(input_name, layers[i + 1])

    return LayeredNeurons(layers)


def LoopBack(layers, input_name=None, synapse_function=None):
    """Connect all the layers in a loop back fashion.

    Each layer's output will be connected to their own input
    matching the given name.

    Args:
        input_name: The input to which to connect
        synapse_function: Optional - the function to use to generate
            the synapses

    Returns:
        A Neurons object containing the layers
    """
    for layer in layers:
        layer.set(input_name, layer)

    return LayeredNeurons(layers)
