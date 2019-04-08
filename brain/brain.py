#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from brain import synapses
from brain import snapshot_pb2


class Brain:
    """A class for handling a brain."""

    def __init__(self, neocortex, sensor):
        """Initialize the brain."""
        self._neocortex = neocortex
        self._sensor = sensor

        synapses.InputConnection(sensor, neocortex._layers[0])

    @property
    def num_inputs(self):
        """The number of inputs the brain can receive through its sensors."""
        return len(self._sensor)

    def snapshot(self, snapshot_to_fill=None):
        """Get a snapshot of the brain state."""
        snap = snapshot_to_fill or snapshot_pb2.Snapshot()
        self._sensor.snapshot(snap.sensor)
        self._neocortex.snapshot(snap.cortex)
        return snap
