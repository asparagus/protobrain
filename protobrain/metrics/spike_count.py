#!/usr/bin/python
# -*- coding: utf-8 -*-
from protobrain.metrics import metric


class SpikeCount(metric.Metric):
    """How many times each neuron spiked during the experiment."""

    def __init__(self):
        """Initialize the metric."""
        ...

    def next(self, neurons):
        ...

    def compute(self):
        ...
