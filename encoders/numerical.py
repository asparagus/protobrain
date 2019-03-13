#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for numerical encoders."""
import abc
import math
import numpy as np


class NumericalEncoder(abc.ABC):
    def __init__(self, min, max, length, sparsity=0.02):
        self._min = min
        self._max = max
        self._length = length
        self._sparsity = sparsity

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._length

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def range(self):
        return self._max - self._min

    @abc.abstractmethod
    def encode(self, value):
        raise NotImplementedError()


class FixedRangeEncoder(NumericalEncoder):
    def encode(self, value):
        if not (self.min <= value <= self.max):
            raise ValueError("Value outside of the range of the encoder.")

        signal_length = math.ceil(self.sparsity * self.length)
        signal_range = self.length - signal_length
        signal_start = int((value - self.min) / self.range * signal_range)

        encoded_value = np.zeros(self.length)
        encoded_value[signal_start:signal_start + signal_length] = 1

        return encoded_value


class CyclicEncoder(NumericalEncoder):
    def encode(self, value):
        if not (self.min <= value <= self.max):
            raise ValueError("Value outside of the range of the encoder.")

        signal_length = math.ceil(self.sparsity * self.length)
        signal_range = self.length - 1
        signal_start = int((value - self.min) / self.range * signal_range)

        encoded_value = np.zeros(self.length)
        encoded_value[signal_start:signal_start + signal_length] = 1
        if signal_start + signal_length > self.length:
            spill_size = (signal_start + signal_length) % self.length
            encoded_value[:spill_size] = 1

        return encoded_value
