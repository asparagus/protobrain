#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Module for implementation of the SpikeCount metric."""
import numpy as np
from protobrain import neuron
from protobrain.metrics import metric


class SpikeCount(metric.Metric):
    """How many times each neuron spiked during the experiment."""

    def __init__(self):
        """Initialize the metric."""
        super().__init__('spike_count')

    def reset(self):
        """Reset all accumulators."""
        self._accumulator = None

    def next(self, neurons):
        """Record the next state for the metric computation.

        Args:
            neurons: Brain state to record
        """
        self._accumulator = self._accumulate_partials(self._accumulator, neurons)

    def _accumulate_partials(self, accumulator, neurons):
        if not isinstance(neurons, neuron.LayeredNeurons):
            return np.array(
                neurons.output.values +
                (accumulator if accumulator is not None else 0))

        if not accumulator:
            accumulator = [None] * len(neurons.layers)

        for i, sublayers in enumerate(neurons.layers):
            accumulator[i] = self._accumulate_partials(
                accumulator[i], sublayers)

        return accumulator

    def _count(self, accumulator):
        if isinstance(accumulator, list):
            return [self._count(subacc) for subacc in accumulator]

        non_zero_indices = list(zip(*np.where(accumulator > 0)))
        counts = {
            0: accumulator.size - len(non_zero_indices),
        }
        for indices in non_zero_indices:
            value = accumulator[indices]
            counts[value] = counts.get(value, 0) + 1

        return counts

    def _aggregate(self, counts):
        if not counts:
            return {}

        if isinstance(counts, dict):
            return dict(counts)

        aggregated_counts = {}
        for subs in counts:
            aggregated_subs = self._aggregate(subs)
            for value, sub_count in aggregated_subs.items():
                aggregated_counts[value] = aggregated_counts.get(value, 0) + sub_count

        return aggregated_counts

    def compute(self):
        """Compute the metric based on the saved state."""
        if self._accumulator is None:
            raise RuntimeError('No iterations - cannot compute metric')
        counts = self._count(self._accumulator)
        return metric.MetricResults(
            self.name,
            global_result=self._aggregate(counts),
            per_layer_result=counts,
        )
