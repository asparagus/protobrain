#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from protobrain import synapses
from protobrain.proto import snapshot_pb2


class ProtoBrain:
    """A class for handling a protobrain."""

    def __init__(self, neocortex, sensor):
        """Initialize the protobrain."""
        self._neocortex = neocortex
        self._sensor = sensor

        synapses.InputConnection(sensor, neocortex._layers[0])

    def snapshot(self, snapshot_to_fill=None):
        """Get a snapshot of the protobrain state."""
        snap = snapshot_to_fill or snapshot_pb2.Snapshot()
        self._sensor.snapshot(snap.sensor)
        self._neocortex.snapshot(snap.cortex)
        return snap
