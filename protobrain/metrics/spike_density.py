#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from protobrain import neuron
from protobrain.metrics import metric


class SpikeDensity(metric.Metric):
    """Fraction of neurons spiked during the experiment."""

    def __init__(self):
        """Initialize the metric."""
        super().__init__('spike_density')
        self._density_per_step_per_layer = []
        self._sizes_per_layer = None

    def next(self, neurons):
        if not self._sizes_per_layer:
            self._sizes_per_layer = self._size_per_layer(neurons)

        self._density_per_step_per_layer.append(
            self._spike_density_per_layer(neurons, self._sizes_per_layer)
        )

    def _size_per_layer(self, layer):
        if isinstance(layer, neuron.LayeredNeurons):
            return tuple(self._size_per_layer(l)
                         for l in layer.layers)
        size = 1.
        for dim in layer.shape:
            size *= dim

        return size

    def _spike_density_per_layer(self, layer, size):
        if isinstance(layer, neuron.LayeredNeurons):
            return tuple(self._spike_density_per_layer(l, size[i])
                         for i, l in enumerate(layer.layers))

        spike_density = np.sum(layer.output.values) / size
        return spike_density

    def _aggregate_over_time(self):
        if isinstance(self._sizes_per_layer, int):
            return (np.sum(self._density_per_step_per_layer) /
                    len(self._density_per_step_per_layer))

        num_steps = len(self._density_per_step_per_layer)
        if num_steps == 1:
            return self._density_per_step_per_layer

        accumulated_layer_density = None
        for layer_density in self._density_per_step_per_layer:
            accumulated_layer_density = self._accumulate_partials(
                accumulated_layer_density, layer_density, num_steps)

        return accumulated_layer_density

    def _accumulate_partials(self, accumulator, layer_density, normalization_factor):
        if isinstance(layer_density, (int, float)):
            return layer_density / normalization_factor + (accumulator or 0)

        if not accumulator:
            accumulator = [None] * len(layer_density)

        for i, sublayer_density in enumerate(layer_density):
            accumulator[i] = self._accumulate_partials(
                accumulator[i], sublayer_density, normalization_factor
            )

        return accumulator

    def _aggregate_over_layers(self):
        if isinstance(self._sizes_per_layer, int):
            return self._density_per_step_per_layer

        num_steps = len(self._density_per_step_per_layer)

        per_step_density = [self._average_density(layer_density, self._sizes_per_layer)[0]
                            for layer_density in self._density_per_step_per_layer]

        return per_step_density

    def _aggregate(self):
        layer_aggregation = self._aggregate_over_layers()
        return np.sum(layer_aggregation) / len(layer_aggregation)

    def _average_density(self, layer_density, size):
        if isinstance(layer_density, tuple):
            weighted_density = 0.
            total_weight = 0
            for i, sublayer_density in enumerate(layer_density):
                d, w = self._average_density(sublayer_density, size[i])
                weighted_density += d * w
                total_weight += w

            return weighted_density / total_weight, total_weight

        return layer_density, size

    def compute(self):
        if not self._sizes_per_layer:
            raise RuntimeError('No iterations - cannot compute metric')
        return metric.MetricResults(
            self.name,
            global_result=self._aggregate(),
            per_layer_result=self._aggregate_over_time(),
            per_step_result=self._aggregate_over_layers(),
            per_layer_per_step_result=self._density_per_step_per_layer,
        )
