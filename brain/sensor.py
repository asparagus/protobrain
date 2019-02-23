#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module for handling a sensory layer.

A neuron is the atomic unit of computations in the brain.
"""
import numpy as np
from brain.event import Event


class Sensor:
    """A class for handling input data."""

    def __init__(self, num_values):
        """Initialize the Sensor."""
        self._values = np.zeros(num_values)
        self._emit = Event()

    def set_values(self, values):
        """Set the new values and propagate them through the emit event."""
        self._values[:] = values
        copy = np.array(self._values)
        self.emit(copy)

    @property
    def emit(self):
        """Get the event for subscribing to the input stream."""
        return self._emit

    @property
    def num_values(self):
        return len(self._values)


# class SensorGroup(Sensor):
#     """A class for handling groups of sensors."""

#     def __init__(self, sensors):
#         super().__init__(sum(s.num_values for s in sensors))
#         self._sensors = sensors
