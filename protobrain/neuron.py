#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module dealing with Neurons."""
import numpy as np
import logging
from protobrain import computation
from protobrain import synapses


log = logging.getLogger(__name__)


class Neurons(object):
    """Class representing neurons."""

    MAIN_INPUT = 'main'

    def __init__(self, shape, computation):
        """Initialize the neurons.

        Args:
            units: Either a number of internal units or a list of layers
            computation: The computation function to use to obtain the outputs
        """
        self.inputs = {
            self.MAIN_INPUT: synapses.Input(self.MAIN_INPUT, shape=shape)
        }
        self.output = synapses.Output(shape=shape)
        self.computation = computation

    def compute(self):
        """Compute the output of this neuron."""
        self.output.values = self.computation(**self.inputs)
        return self.values

    @property
    def input(self):
        """Get the main input."""
        return self.get(Neurons.MAIN_INPUT)

    @input.setter
    def input(self, output):
        """Connect the main input to the given output."""
        self.set(Neurons.MAIN_INPUT, output)

    def get(self, name):
        """Get an input to these neurons by name."""
        if name not in self.inputs:
            raise IndexError(
                '{0} not set as an input for {1}'.format(
                    name, repr(self)
                ))
        return self.inputs[name]

    def set(self, name, output):
        """Connect an input to the given output."""
        self.inputs[name] = synapses.Input(name, shape=self.output.shape)
        self.inputs[name].connect(output)

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
            log.warning(
                'Creating layers with pre-connected input'
            )

    def compute(self):
        """Compute the output of these layers."""
        for layer in self.layers:
            layer.compute()
        return self.values


def FeedForward(layers, input_name='main'):
    """Connect all the layers in a feed forward fashion.

    Each layer's output will be connected to the next layer's input
    matching the given name.

    Args:
        input_name: The input to which to connect

    Returns:
        A Neurons object containing the layers
    """
    for i, layer in enumerate(layers[:-1]):
        layers[i + 1].set(input_name, layer)

    return LayeredNeurons(layers)


def FeedBackward(layers, input_name=None):
    """Connect all the layers in a feed backward fashion.

    Each layer's output will be connected to the previous layer's input
    matching the given name.

    Args:
        input_name: The input to which to connect

    Returns:
        A Neurons object containing the layers
    """
    for i, layer in enumerate(layers[:-1]):
        layer.set(input_name, layers[i + 1])

    return LayeredNeurons(layers)


def LoopBack(layers, input_name=None):
    """Connect all the layers in a loop back fashion.

    Each layer's output will be connected to their own input
    matching the given name.

    Args:
        input_name: The input to which to connect

    Returns:
        A Neurons object containing the layers
    """
    for layer in layers:
        layer.set(input_name, layer)

    return LayeredNeurons(layers)
