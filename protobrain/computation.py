#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import numpy as np


class Computation(abc.ABC):

    @abc.abstractmethod
    def __call__(self, inputs, feedback, inhibitions):
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__


class StandardComputation(Computation):
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, main):
        """
        Compute the neurons' output.

        Ignores feedback and inhibitions for now.
        """
        activations = np.dot(main.synapses, main.values)
        return activations > self.threshold

    @property
    def threshold(self):
        return self._threshold


class SparseComputation(Computation):
    def __init__(self, n):
        self._n = n

    def __call__(self, main):
        """
        Compute the neurons' output.

        Ignores feedback and inhibitions for now.
        """
        activations = np.dot(main.synapses, main.values)
        top_indices = activations.argsort()[-self._n:]

        result = np.zeros(len(activations))
        result[top_indices] = 1
        return result

    @property
    def n(self):
        return self._n

