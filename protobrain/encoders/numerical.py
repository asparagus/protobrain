#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for numerical encoders."""
import math
import numpy as np
from protobrain import sensor


class NumericalEncoder(sensor.Encoder):
    def __init__(self, min_value, max_value, length, sparsity=0.02):
        super().__init__(
            default_value=0,
            shape=(length,)
        )
        self.min = min_value
        self.max = max_value
        self._length = length
        self._sparsity = sparsity

    @property
    def range(self):
        return self.max - self.min

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def length(self):
        return self._length


class SimpleEncoder(NumericalEncoder):
    def encode(self, value):
        if not (self.min <= value <= self.max):
            raise ValueError('Value outside of the range of the encoder.')

        signal_length = math.ceil(self.sparsity * self.length)
        signal_range = self.length - signal_length
        signal_start = int((value - self.min) / self.range * signal_range)

        encoded_value = np.zeros(self.length)
        encoded_value[signal_start:signal_start + signal_length] = 1

        return encoded_value


class CyclicEncoder(NumericalEncoder):
    def encode(self, value):
        if not (self.min <= value <= self.max):
            raise ValueError('Value outside of the range of the encoder.')

        signal_length = math.ceil(self.sparsity * self.length)
        signal_range = self.length - 1
        signal_start = int((value - self.min) / self.range * signal_range)

        encoded_value = np.zeros(self.length)
        encoded_value[signal_start:signal_start + signal_length] = 1
        if signal_start + signal_length > self.length:
            spill_size = (signal_start + signal_length) % self.length
            encoded_value[:spill_size] = 1

        return encoded_value
