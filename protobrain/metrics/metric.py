#!/usr/bin/python
# -*- coding: utf-8 -*-
import abc
from protobrain import neuron


class MetricResults(object):
    def __init__(self, metric_name, *,
                 global_result=None,
                 per_layer_result=None,
                 per_step_result=None,
                 per_layer_per_step_result=None):
        self.metric_name = metric_name
        self.global_result = global_result
        self.per_layer_result = per_layer_result
        self.per_step_result = per_step_result
        self.per_layer_per_step_result = per_layer_per_step_result

    def __str__(self):
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
        """Compute the metric."""
        raise NotImplementedError()

    @abc.abstractmethod
    def next(self, neurons):
        """Pass the next state for the metric computation."""
        raise NotImplementedError()

    @property
    def name(self):
        """Gets the name of the metric."""
        return self._name
