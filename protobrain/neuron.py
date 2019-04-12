#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module dealing with Neurons."""
import numpy as np
from protobrain import computation
from protobrain import synapses


class Neurons(object):
    MAIN_INPUT = 'main'

    def __init__(self, units, comp=None):
        self._passthrough = isinstance(units, list)
        if self._passthrough:
            self._layers = units
            # TODO warn if set up seems wrong
            self._inputs = units[0]._inputs
            self.output = units[-1].output
        else:
            self._inputs = {
                Neurons.MAIN_INPUT: synapses.Input(
                    Neurons.MAIN_INPUT,
                    shape=(units,)
                )
            }
            self.output = synapses.Output(shape=(units,))

        self._computation = comp

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
        if name not in self._inputs:
            raise IndexError(
                '{0} not set as an input for {1}'.format(
                    name, repr(self)
                ))
        return self._inputs[name]

    def set(self, name, output):
        """Connect an input to the given output."""
        if name not in self._inputs:
            self._inputs[name] = synapses.Input(name, shape=self.input.shape)

        self._inputs[name].connect(output)

    def compute(self, computation=None):
        """Compute the output of this neuron."""
        computation_function = computation or self._computation
        if self._passthrough:
            for layer in self._layers:
                layer.compute(computation_function)
        else:
            self.output.values = computation_function(**self._inputs)

    @property
    def values(self):
        """The values at the output of these neurons."""
        return self.output.values

    @property
    def shape(self):
        """The shape of the output of these neurons."""
        return self.output.shape


def FeedForward(layers, input_name='main'):
    for i, layer in enumerate(layers[:-1]):
        layers[i + 1].set(input_name, layer)

    return Neurons(layers)


def FeedBackward(layers, input_name=None):
    for i, layer in enumerate(layers[:-1]):
        layer.set(input_name, layers[i + 1])

    return Neurons(layers)


def LoopBack(layers, input_name=None):
    for layer in layers:
        layer.set(input_name, layer)

    return Neurons(layers)
