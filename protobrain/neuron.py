#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module dealing with Neurons."""
import numpy as np
from protobrain import event
from protobrain.util import factory


class Neurons:
    def __init__(self, number, computation):
        self._number = number
        self._computation = computation
        self._outputs = None
        self._inputs = None
        self._feedback = None
        self._inhibitions = None
        self._emit = event.Event()

    def compute(self):
        self._outputs = self._computation(self._inputs, self._feedback, self._inhibitions)
        self.emit(self._outputs)

    def set_inputs(self, inputs):
        self._inputs = inputs

    def set_feedback(self, feedback):
        self._feedback = feedback

    def set_inhibitions(self, inhibitions):
        self._inhibitions = inhibitions

    @property
    def outputs(self):
        return self._outputs

    @property
    def emit(self):
        return self._emit

    def __len__(self):
        return self._number

    def __str__(self):
        return str(self._outputs.astype(np.int32))


class NeuronInput:
    def __init__(self, values, synaptic_strength):
        self._values = values
        self._synaptic_strength = synaptic_strength

    @property
    def synaptic_strength(self):
        return self._synaptic_strength

    @property
    def values(self):
        return self._values


class NeuronsFactory(factory.Factory):
    """A factory class for Neurons."""

    def __init__(self, number_factory, computation_factory):
        """Initialize a Neuron Factory."""
        self._number_factory = number_factory
        self._computation_factory = computation_factory

    def create(self):
        """Create Neurons."""
        return Neurons(
            self._number_factory(),
            self._computation_factory())


class SimpleNeuronsFactory(NeuronsFactory):
    """A simple factory class for Neurons."""

    def __init__(self, number, computation):
        """Initialize the factory."""
        super().__init__(
            factory.ConstantFactory(number),
            factory.ConstantFactory(computation)
        )
