#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for handling how synapses are formed."""
import abc


class SynapticPolicy(abc.ABC):

    @abc.abstractmethod
    def execute(self, layers):
        raise NotImplementedError()

class NullSynapticPolicy(abc.ABC):
    def execute(self, layers):
        return []

class ForwardSynapticPolicy(SynapticPolicy):

    def __init__(self, synapse_creation):
        self._synapse_creation = synapse_creation

    def execute(self, layers):
        connections = [
            self._synapse_creation(layers[i - 1], layers[i])
            for i in range(1, len(layers))
        ]

        return connections

class BackwardSynapticPolicy(SynapticPolicy):

    def __init__(self, synapse_creation):
        self._synapse_creation = synapse_creation

    def execute(self, layers):
        connections = [
            self._synapse_creation(layers[i], layers[i - 1])
            for i in range(1, len(layers))
        ]

        return connections

class ReflexiveSynapticPolicy(SynapticPolicy):

    def __init__(self, synapse_creation):
        self._synapse_creation = synapse_creation

    def execute(self, layers):
        connections = [self._synapse_creation(layer)
                       for layer in layers]

        return connections

class SpecificSynapticPolicy(SynapticPolicy):
    def __init__(self, synapse_creation, specification):
        self._synapse_creation = synapse_creation
        self._specification = specification

    def execute(self, layers):
        connections = []
        for idx, indices in self._specification.items():
            if not isinstance(indices, list):
                indices = [indices]

            for i in indices:
                connections.append(
                    self._synapse_creation(layers[idx], layers[i])
                )

        return connections
