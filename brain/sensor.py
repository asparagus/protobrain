#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for handling a sensory layer."""
import numpy as np
from brain import event


class Sensor:
    """A class for handling input data."""

    def __init__(self, num_values):
        """Initialize the Sensor."""
        self._values = np.zeros(num_values)
        self._emit = event.Event()

    def set_values(self, values):
        """Set the new values and propagate them through the emit event."""
        if len(values) != len(self):
            raise ValueError("Sensor values length do not match specification")
        self._values = values
        self.emit(values)

    @property
    def emit(self):
        """Get the event for subscribing to the input stream."""
        return self._emit

    def __len__(self):
        """Get the number of values returned by this sensor."""
        return len(self._values)

    def __str__(self):
        """The string representation of this sensor is its values."""
        return "%s: %s" % (self.__class__, self._values)
