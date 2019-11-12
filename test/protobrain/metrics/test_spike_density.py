#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from protobrain import neuron
from protobrain.metrics import spike_density


@pytest.fixture(scope='function')
def neurons():
    return neuron.Neurons(4, None)

@pytest.fixture(scope='function')
def layers():
    return neuron.FeedForward([neuron.Neurons(i, None)
                               for i in (4, 4, 4, 4)])

@pytest.fixture(scope='function')
def non_uniform_layers(layers, neurons):
    return neuron.FeedForward([
        layers,
        neurons,
    ])

def test_no_timesteps(neurons):
    metric = spike_density.SpikeDensity()

    with pytest.raises(RuntimeError):
        result = metric.compute()

def test_constant_density_neurons(neurons):
    metric = spike_density.SpikeDensity()

    neurons.output.values = np.array([1, 0, 0, 0])
    metric.next(neurons)
    metric.next(neurons)

    result = metric.compute()
    assert result.metric_name == 'spike_density'
    assert result.global_result == 0.25
    assert result.per_layer_result == 0.25
    assert result.per_step_result == [0.25, 0.25]
    assert result.per_layer_per_step_result == [0.25, 0.25]

def test_constant_density_layers(layers):
    metric = spike_density.SpikeDensity()

    for sublayer in layers.layers:
        sublayer.output.values = np.array([1, 0, 0, 0])

    metric.next(layers)
    metric.next(layers)

    result = metric.compute()
    assert result.metric_name == 'spike_density'
    assert result.global_result == 0.25
    assert result.per_layer_result == [0.25, 0.25, 0.25, 0.25]
    assert result.per_step_result == [0.25, 0.25]
    assert result.per_layer_per_step_result == [(0.25, 0.25, 0.25, 0.25),
                                                (0.25, 0.25, 0.25, 0.25)]

def test_constant_non_uniform_layers(non_uniform_layers):
    metric = spike_density.SpikeDensity()

    for sublayer in non_uniform_layers.layers[0].layers:
        sublayer.output.values = np.array([1, 0, 0, 0])
    for sublayer in non_uniform_layers.layers[1:]:
        sublayer.output.values = np.array([1, 0, 0, 0])

    metric.next(non_uniform_layers)
    metric.next(non_uniform_layers)

    result = metric.compute()
    assert result.metric_name == 'spike_density'
    assert result.global_result == 0.25
    assert result.per_layer_result == [[0.25, 0.25, 0.25, 0.25], 0.25]
    assert result.per_step_result == [0.25, 0.25]
    assert result.per_layer_per_step_result == [
        ((0.25, 0.25, 0.25, 0.25), 0.25),
        ((0.25, 0.25, 0.25, 0.25), 0.25)]

def test_changing_density_neurons(neurons):
    metric = spike_density.SpikeDensity()

    neurons.output.values = np.array([1, 0, 0, 0])
    metric.next(neurons)

    neurons.output.values = np.array([1, 1, 0, 0])
    metric.next(neurons)

    result = metric.compute()
    assert result.metric_name == 'spike_density'
    assert result.global_result == pytest.approx(0.375)
    assert result.per_layer_result == 0.375
    assert result.per_step_result == [0.25, 0.50]
    assert result.per_layer_per_step_result == [0.25, 0.50]

def test_changing_density_layers(layers):
    metric = spike_density.SpikeDensity()

    for sublayer in layers.layers:
        sublayer.output.values = np.array([1, 0, 0, 0])
    metric.next(layers)

    for sublayer in layers.layers:
        sublayer.output.values = np.array([1, 1, 0, 0])
    metric.next(layers)

    result = metric.compute()
    assert result.metric_name == 'spike_density'
    assert result.global_result == pytest.approx(0.375)
    assert result.per_layer_result == [0.375, 0.375, 0.375, 0.375]
    assert result.per_step_result == [0.25, 0.50]
    assert result.per_layer_per_step_result == [(0.25, 0.25, 0.25, 0.25),
                                                (0.50, 0.50, 0.50, 0.50)]

def test_changing_density_non_uniform_layers(non_uniform_layers):
    metric = spike_density.SpikeDensity()

    for sublayer in non_uniform_layers.layers[0].layers:
        sublayer.output.values = np.array([1, 0, 0, 0])
    for sublayer in non_uniform_layers.layers[1:]:
        sublayer.output.values = np.array([1, 1, 0, 0])
    metric.next(non_uniform_layers)

    for sublayer in non_uniform_layers.layers[0].layers:
        sublayer.output.values = np.array([1, 0, 1, 0])
    for sublayer in non_uniform_layers.layers[1:]:
        sublayer.output.values = np.array([1, 1, 0, 1])
    metric.next(non_uniform_layers)

    result = metric.compute()
    assert result.metric_name == 'spike_density'
    assert result.global_result == pytest.approx(0.425)
    assert result.per_layer_result == [[0.375, 0.375, 0.375, 0.375], 0.625]
    assert result.per_step_result == [0.3, 0.55]
    assert result.per_layer_per_step_result == [
        ((0.25, 0.25, 0.25, 0.25), 0.50),
        ((0.50, 0.50, 0.50, 0.50), 0.75)]
