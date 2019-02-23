#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for modelling the Neocortex.

MiniColumns are a grouping of neurons that are connected to the same inputs.
"""
import numpy as np
from brain.layer import SimpleLayerFactory
from util.factory import Factory, ConstantFactory


class Neocortex:
    """The class for the Neocortex."""

    def __init__(self, layers):
        """Initialize the neocortex."""
        assert len(layers) > 0
        self._layers = layers
        for i, layer in enumerate(self._layers[1:]):
            layer.connect(self._layers[i])
        self._learning = True

    def process(self):
        """Process through all the neocortex."""
        for layer in self._layers:
            layer.process()

        self.learn()

    def learn(self):
        """Do some learning."""
        pass

    @property
    def learning(self):
        """Whether the neocortex is in learning mode."""
        return self._learning

    @property
    def num_layers(self):
        """The number of layers in the neocortex."""
        return len(self._layers)


class NeocortexFactory(Factory):
    """A Factory class for the Neocortex."""

    def __init__(self, num_layers_factory, layer_factory):
        """Initialize the factory."""
        self._num_layers_factory = num_layers_factory
        self._layer_factory = layer_factory

    def create(self):
        """Create a neocortex."""
        num_layers = self._num_layers_factory()
        layers = [self._layer_factory()
                   for _ in range(num_layers)]
        return Neocortex(layers)


class SimpleNeocortexFactory(NeocortexFactory):
    """A simple factory class for the Neocortex."""

    def __init__(self,
                 num_layers,
                 num_minicolumns,
                 num_neurons,
                 num_synapses,
                 neuron_threshold=0.5,
                 synapse_threshold=0.5):
        """Initialize the factory."""
        num_layers_factory = ConstantFactory(num_layers)
        layer_factory = SimpleLayerFactory(
            num_minicolumns, num_neurons, num_synapses,
            neuron_threshold, synapse_threshold
        )

        super().__init__(num_layers_factory, layer_factory)
