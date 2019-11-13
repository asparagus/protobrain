#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
from protobrain import neuron


class MetricResults(object):
    """Results from a metric evaluation."""

    def __init__(self, metric_name, *,
                 global_result=None,
                 per_layer_result=None,
                 per_step_result=None,
                 per_layer_per_step_result=None):
        """Initialize the results.

        Results require a metric's name and optionally allow setting values at
        different levels of aggregation.

        Args:
            metric_name: The name of the metric.
            global_result: Optional - the aggregated result
            per_layer_result: Optional - the aggregated result to a layer level
            per_step_result: Optional- the aggregated result to a step level
            per_layer_per_step_result: Optional - the computed result per
                layer / step combination
        """
        self.metric_name = metric_name
        self.global_result = global_result
        self.per_layer_result = per_layer_result
        self.per_step_result = per_step_result
        self.per_layer_per_step_result = per_layer_per_step_result

    def __str__(self):
        """Get the string representation of this metric."""
        data = ['%s:' % self.metric_name]
        if self.global_result:
            data.append('Global: %s' % self.global_result)
        if self.per_layer_result:
            data.append('Per layer:\n%s' % self.per_layer_result)
        if self.per_step_result:
            data.append('Per step:\n%s' % self.per_step_result)
        if self.per_layer_per_step_result:
            data.append('Per layer - per step:\n%s' % self.per_layer_per_step_result)

        return '\n'.join(data)


class Metric(abc.ABC):
    """A class to represent a metric for an experiment."""

    def __init__(self, name):
        """Initialize the metric with a name."""
        self._name = name

    @abc.abstractmethod
    def compute(self):
        """Compute the metric based on the saved state."""
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self, neurons):
        """Record the next state for the metric computation.

        Args:
            neurons: Brain state to record
        """
        raise NotImplementedError()

    @property
    def name(self):
        """Gets the name of the metric."""
        return self._name
