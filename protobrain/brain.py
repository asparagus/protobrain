#!/usr/bin/python
# -*- coding: utf-8 -*-
# from protobrain.proto import snapshot_pb2


class Brain:
    """A class for handling a protobrain."""

    def __init__(self, neurons, sensor):
        """Initialize the protobrain."""
        self._neurons = neurons
        self._sensor = sensor

        neurons.input = sensor

    def compute(self, computation_function=None):
        self._neurons.compute(computation_function)

    def learn(self, learning_function=None):
        learning_function(self._neurons)
    # def snapshot(self, snapshot_to_fill=None):
    #     """Get a snapshot of the protobrain state."""
    #     snap = snapshot_to_fill or snapshot_pb2.Snapshot()
    #     self._sensor.snapshot(snap.sensor)
    #     self._neurons.snapshot(snap.cortex)
    #     return snap
