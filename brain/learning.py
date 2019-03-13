#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
import numpy as np


class Learning(abc.ABC):

    @abc.abstractmethod
    def __call__(self, connections):
        raise NotImplementedError()


class HebbianLearning(abc.ABC):
    def __init__(self, increase, decrease):
        self._increase = increase
        self._decrease = decrease

    def __call__(self, connections):
        for connection in connections:
            m = connection._other_neurons.outputs[..., np.newaxis]
            n = connection._neurons.outputs[np.newaxis, ...]

            matches = np.dot(m, n)

            # Strengthen matching connections
            connection._strength += matches * self._increase

            # Weaken all connections
            connection._strength -= self._decrease

            # Ensure the values remain within valid range
            connection._strength = np.clip(connection._strength, 0, 1)
