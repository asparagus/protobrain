#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for handling Neurons.

A neuron is the atomic unit of computations in the brain.
"""
import json
import numpy as np
from util.factory import Factory, ConstantFactory


class Neuron:
    """Class for implementing the Neuron."""

    def __init__(self, threshold, synapse_threshold, synaptic_permanence):
        """Initialize the neuron."""
        self._threshold = threshold
        self._synapse_threshold = synapse_threshold
        self._synaptic_permanence = synaptic_permanence

    def compute(self, inputs):
        """
        Obtain the output for this neuron.

        First filters the inputs according to whether they pass
        the synapse threshold, then checks the percentage of active synapses
        against the neuron threshold.
        """
        active_synapses = self._synaptic_permanence >= self._synapse_threshold
        return np.dot(active_synapses, inputs) / len(inputs) > self.threshold

    @property
    def num_synapses(self):
        """The number of synapses in this neuron."""
        return len(self._synaptic_permanence)

    @property
    def synaptic_permanence(self):
        """Get the values of the synapses for this neuron."""
        return np.copy(self._synaptic_permanence)

    @property
    def threshold(self):
        """Get the threshold value for this neuron."""
        return self._threshold

    def __str__(self):
        """String representation of this neuron."""
        return "%s: %s" % (self.__class__, json.dumps({
            "threshold": self._threshold,
            "synapse_threshold": self._synapse_threshold,
            "synaptic_permanence": str(self._synaptic_permanence)
        }, indent=4))


class NeuronFactory(Factory):
    """A factory class for Neurons."""

    def __init__(self, threshold_factory, synapse_threshold_factory, synaptic_permanence_factory):
        """Initialize a Neuron Factory."""
        self._threshold_factory = threshold_factory
        self._synapse_threshold_factory = synapse_threshold_factory
        self._synaptic_permanence_factory = synaptic_permanence_factory

    def create(self):
        """Create a Neuron."""
        return Neuron(
            self._threshold_factory(),
            self._synapse_threshold_factory(),
            self._synaptic_permanence_factory())


class SimpleNeuronFactory(NeuronFactory):
    """A simple factory class for Neurons."""

    def __init__(self, num_synapses, threshold=None, synapse_threshold=None):
        """Initialize the factory."""
        if threshold is None:
            threshold_factory = np.random.rand
        else:
            threshold_factory = lambda: threshold

        if synapse_threshold is None:
            synapse_threshold_factory = np.random.rand
        else:
            synapse_threshold_factory = lambda: synapse_threshold

        super().__init__(
            threshold_factory,
            synapse_threshold_factory,
            lambda: np.random.rand(num_synapses)
        )
