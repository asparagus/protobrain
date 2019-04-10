#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for handling neuron connections."""
import abc
import numpy as np
from protobrain import neuron


class Synapses(abc.ABC):

    def __init__(
            self, neurons, other_neurons=None, sparsity=0.02):
        self._neurons = neurons
        self._other_neurons = other_neurons or neurons
        self._sparsity = sparsity
        self._mask, self._strength =\
            Synapses._create_connections(
                self._neurons,
                self._other_neurons,
                self._sparsity
            )

        neurons.emit.subscribe(lambda values:
            self.propagate(neuron.NeuronInput(values, self.strength)))

    @abc.abstractmethod
    def propagate(self, arg):
        raise NotImplementedError()

    @property
    def mask(self):
        return self._mask

    @property
    def strength(self):
        return self._strength

    @staticmethod
    def _create_connections(inp, out, sparsity):
        shape = len(out), len(inp)
        if inp is out:  # Self connections
            mask_generator = Synapses._random_symmetric_mask_for_connections
        else:
            mask_generator = Synapses._random_mask_for_connections

        mask = mask_generator(shape, sparsity)
        strength = np.random.uniform(0, 1, mask.shape)
        strength[mask] = 0

        return mask, strength

    @staticmethod
    def _random_mask_for_connections(shape, sparsity):
        random = np.random.uniform(0, 1, shape)
        return random < sparsity

    @staticmethod
    def _random_symmetric_mask_for_connections(shape, sparsity):
        random = np.random.uniform(0, 1, shape)
        symmetric = (random + random.T) / 2
        mask = symmetric < sparsity
        np.fill_diagonal(mask, False)
        return mask

class InputConnection(Synapses):

    def propagate(self, values):
        self._other_neurons.set_inputs(values)

class FeedbackConnection(Synapses):

    def propagate(self, values):
        self._other_neurons.set_feedback(values)

class InhibitionConnection(Synapses):

    def propagate(self, values):
        self._other_neurons.set_inhibitions(values)
