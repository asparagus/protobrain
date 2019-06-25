#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc


class Metric(abc.ABC):
    """A class to represent a metric for an experiment."""

    def __init__(self, name):
        """Initialize the metric with a name."""
        self._name = name

    @abc.abstractmethod
    def compute(self):
        """Compute the metric."""
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self, state):
        """Pass the next state for the metric computation."""
        raise NotImplementedError()

    @property
    def name(self):
        return self._name


class Coverage(Metric):
    """How many neurons spiked during the experiment."""

    def __init__(self):
        """Initialize the metric."""
        super().__init__('coverage')
        self._tracker = None
        self._coverage_per_step = []
        self._coverage_per_step_per_layer = []

    def next(self, snapshot):
        on_bits_by_layer = [
            set(sdr.on_bits)
            for sdr in snapshot.cortex
        ]

        if self._tracker is None:
            self._tracker = on_bits_by_layer
        else:
            for i, layer in enumerate(on_bits_by_layer):
                self._tracker[i] |= layer

    def compute(self):
        """Compute the metric."""
        neurons_that_turned_on = sum(float(neuron) for neuron in self._tracker)
        return neurons_that_turned_on / len(self._tracker)
