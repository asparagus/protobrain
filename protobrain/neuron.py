"""Module dealing with Neurons."""

import logging
from typing import Callable

import numpy as np

from protobrain import computation as _computation
from protobrain import learning as _learning
from protobrain import synapses


log = logging.getLogger(__name__)


class Neurons:
    """Class representing neurons."""

    MAIN_INPUT = "main"

    def __init__(
        self,
        shape: tuple[int, ...] | int,
        computation: _computation.Computation | None = None,
        learning: _learning.Learning | None = None,
    ):
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

    def compute(self) -> np.ndarray:
        """Compute the output of this neuron."""
        if not self.computation:
            raise ValueError(f"Computation function not set for {self}!")
        self.output.values = self.computation(**self.inputs)
        return self.values

    def learn(self) -> None:
        """Adjust the synapses to learn."""
        if not self.learning:
            raise ValueError(f"Learning function not set for {self}!")
        if self.learning:
            self.learning(self)

    @property
    def input(self) -> synapses.Input:
        """Get the main input."""
        return self.get(Neurons.MAIN_INPUT)

    @input.setter
    def input(self, output: synapses.Output):
        """Connect the main input to the given output."""
        self.set(Neurons.MAIN_INPUT, output)

    def get(self, name: str) -> synapses.Input:
        """Get an input to these neurons by name.

        Args:
            name: The name of the input to retrieve

        Returns:
            The input that goes by that name.
        """
        if name not in self.inputs:
            raise IndexError("{0} not set as an input for {1}".format(name, repr(self)))
        return self.inputs[name]

    def set(
        self,
        name: str,
        output: synapses.Output,
        synapse_function: Callable[[tuple[int, ...], tuple[int, ...]], None] = None,
    ):
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
    def values(self) -> np.ndarray:
        """The values at the output of these neurons."""
        return self.output.values

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the output of these neurons."""
        return self.output.shape


class LayeredNeurons(Neurons):
    """Class representing groups of neurons."""

    def __init__(self, layers: list[Neurons]):
        self.layers = layers
        self.inputs = self.layers[0].inputs
        self.output = self.layers[-1].output

        if self.inputs[self.MAIN_INPUT].connected:
            log.warning("Creating layers with pre-connected input")

    def learn(self) -> None:
        """Adjust the synapses to learn."""
        for layer in self.layers:
            layer.learn()

    def compute(self) -> np.ndarray:
        """Compute the output of these layers."""
        for layer in self.layers:
            layer.compute()
        return self.values

    @property
    def computation(self) -> list[_computation.Computation]:
        return [layer.computation for layer in self.layers]

    @computation.setter
    def computation(self, function: _computation.Computation):
        for layer in self.layers:
            layer.computation = function

    @property
    def learning(self) -> list[_learning.Learning]:
        return [layer.learning for layer in self.layers]

    @learning.setter
    def learning(self, function: _learning.Learning):
        for layer in self.layers:
            layer.learning = function


def FeedForward(
    layers: list[Neurons],
    input_name: str = "main",
    synapse_function: Callable[[tuple[int, ...], tuple[int, ...]], None] = None,
) -> LayeredNeurons:
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


def FeedBackward(
    layers: list[Neurons],
    input_name: str = None,
    synapse_function: Callable[[tuple[int, ...], tuple[int, ...]], None] = None,
) -> LayeredNeurons:
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


def LoopBack(
    layers: list[Neurons],
    input_name: str = None,
    synapse_function: Callable[[tuple[int, ...], tuple[int, ...]], None] = None,
) -> LayeredNeurons:
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
