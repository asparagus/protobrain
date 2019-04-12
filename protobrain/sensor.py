#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for handling a sensory layer."""
import abc
from protobrain import synapses
# from protobrain.proto import snapshot_pb2
# from protobrain.util import sdr


class Sensor(object):
    """A class for handling input data."""

    def __init__(self, encoder):
        """Initialize the Sensor."""
        self._encoder = encoder
        self._value = encoder.default_value
        self.output = synapses.Output(encoder.shape)

    def feed(self, value):
        """Feed a value to the sensor."""
        self._value = value
        self.output.values = self._encoder.encode(value)

    @property
    def value(self):
        """Get the value encoded by this sensor."""
        return self._value

    @value.setter
    def value(self, output):
        """Set the value encoded by this sensor."""
        self.feed(value)

    @property
    def values(self):
        """The values from the output unit."""
        return self.output.values

    @property
    def shape(self):
        """The shape of this sensor's outputs."""
        return self.output.shape

    # def snapshot(self, snapshot_to_fill):
    #     """Get a snapshot of the sensor state."""
    #     snap = snapshot_to_fill or snapshot_pb2.SensorSnapshot()
    #     sdr.np_to_sdr(self.values, snap.sdr)
    #     return snap


class Encoder(abc.ABC):

    def __init__(self, default_value, shape):
        self.default_value = default_value
        self.shape = shape

    @abc.abstractmethod
    def encode(self, value):
        raise NotImplementedError()
