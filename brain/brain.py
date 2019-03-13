#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from brain.synapses import InputConnection


class Brain:
    """A class for handling a brain."""

    def __init__(self, neocortex, sensor):
        """Initialize the brain."""
        self._neocortex = neocortex
        self._sensor = sensor

        InputConnection(sensor, neocortex._layers[0])

    @property
    def num_inputs(self):
        """The number of inputs the brain can receive through its sensors."""
        return len(self._sensor)
