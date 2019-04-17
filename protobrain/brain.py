#!/usr/bin/python
# -*- coding: utf-8 -*-


class Brain:
    """A class for handling a protobrain."""

    def __init__(self, neurons, sensor):
        """Initialize the protobrain."""
        self.neurons = neurons
        self.sensor = sensor

        neurons.input = sensor

    def compute(self, computation_function=None):
        self.neurons.compute(computation_function)

    def learn(self, learning_function=None):
        learning_function(self.neurons)
