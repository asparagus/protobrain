#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for modelling the Neocortex.

MiniColumns are a grouping of neurons that are connected to the same inputs.
"""
import numpy as np
from brain import neuron
from brain import snapshot_pb2
from util import factory
from util import sdr


class Neocortex:
    """The class for the Neocortex."""

    def __init__(self, layers, synaptic_policies, learning_function=None):
        """Initialize the neocortex."""
        assert len(layers) > 0
        self._layers = layers
        self._connections = [
            connections for policy in synaptic_policies
            for connections in policy.execute(layers)]
        self._learning_function = learning_function

    def process(self):
        """Process through all the neocortex."""
        for layer in self._layers:
            layer.compute()

        if self._learning_function is not None:
            self._learning_function(self._connections)

    @property
    def num_layers(self):
        """The number of layers in the neocortex."""
        return len(self._layers)

    def __str__(self):
        """The string representation of the neocortex's state."""
        return "%s:\n\t%s" % (
            self.__class__,
            "\n\t".join([str(layer) for layer in self._layers]))

    def snapshot(self, snapshot_to_fill):
        """Get a snapshot of the neocortex state."""
        snap = snapshot_to_fill or snapshot_pb2.CorticalSnapshot()
        for layer in self._layers:
            layer_sdr = snap.sdr.add()
            sdr.np_to_sdr(layer.outputs, layer_sdr)
        return snap


class NeocortexFactory(factory.Factory):
    """A Factory class for the Neocortex."""

    def __init__(self,
                 num_layers_factory,
                 neurons_factory,
                 synaptic_policies,
                 learning_function):
        """Initialize the factory."""
        self._num_layers_factory = num_layers_factory
        self._neurons_factory = neurons_factory
        self._synaptic_policies = synaptic_policies
        self._learning_function = learning_function

    def create(self):
        """Create a neocortex."""
        num_layers = self._num_layers_factory()
        layers = [self._neurons_factory() for i in range(num_layers)]
        return Neocortex(layers,
                         self._synaptic_policies,
                         self._learning_function)


class SimpleNeocortexFactory(NeocortexFactory):
    """A simple factory class for the Neocortex."""

    def __init__(self,
                 num_layers,
                 num_neurons,
                 computation,
                 synaptic_policies,
                 learning_function):
        """Initialize the factory."""
        num_layers_factory = factory.ConstantFactory(num_layers)
        neurons_factory = neuron.SimpleNeuronsFactory(
            num_neurons,
            computation
        )

        super().__init__(
            num_layers_factory,
            neurons_factory,
            synaptic_policies,
            learning_function
        )
