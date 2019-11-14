#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from protobrain import neuron
from protobrain.metrics import spike_count


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
    metric = spike_count.SpikeCount()

    with pytest.raises(RuntimeError):
        result = metric.compute()


def test_constant_density_neurons(neurons):
    metric = spike_count.SpikeCount()

    neurons.output.values = np.array([1, 0, 0, 0])
    metric.next(neurons)
    metric.next(neurons)

    result = metric.compute()
    assert result.metric_name == 'spike_count'
    assert result.global_result == {0: 3, 2: 1}
    assert result.per_layer_result == {0: 3, 2: 1}


def test_constant_density_layers(layers):
    metric = spike_count.SpikeCount()

    for sublayer in layers.layers:
        sublayer.output.values = np.array([1, 0, 0, 0])

    metric.next(layers)
    metric.next(layers)

    result = metric.compute()
    assert result.metric_name == 'spike_count'
    assert result.global_result == {0: 12, 2: 4}
    assert result.per_layer_result == [{0: 3, 2: 1},
                                       {0: 3, 2: 1},
                                       {0: 3, 2: 1},
                                       {0: 3, 2: 1}]


def test_constant_non_uniform_layers(non_uniform_layers):
    metric = spike_count.SpikeCount()

    for sublayer in non_uniform_layers.layers[0].layers:
        sublayer.output.values = np.array([1, 0, 0, 0])
    for sublayer in non_uniform_layers.layers[1:]:
        sublayer.output.values = np.array([1, 0, 0, 0])

    metric.next(non_uniform_layers)
    metric.next(non_uniform_layers)

    result = metric.compute()
    assert result.metric_name == 'spike_count'
    assert result.global_result == {0: 15, 2: 5}
    assert result.per_layer_result == [
        [{0: 3, 2: 1},
         {0: 3, 2: 1},
         {0: 3, 2: 1},
         {0: 3, 2: 1}], {0: 3, 2: 1}]


def test_changing_density_neurons(neurons):
    metric = spike_count.SpikeCount()

    neurons.output.values = np.array([1, 0, 0, 0])
    metric.next(neurons)

    neurons.output.values = np.array([1, 1, 0, 0])
    metric.next(neurons)

    result = metric.compute()
    assert result.metric_name == 'spike_count'
    assert result.global_result == {0: 2, 1: 1, 2: 1}
    assert result.per_layer_result == {0: 2, 1: 1, 2: 1}


def test_changing_density_layers(layers):
    metric = spike_count.SpikeCount()

    for sublayer in layers.layers:
        sublayer.output.values = np.array([1, 0, 0, 0])
    metric.next(layers)

    for sublayer in layers.layers:
        sublayer.output.values = np.array([1, 1, 0, 0])
    metric.next(layers)

    result = metric.compute()
    assert result.metric_name == 'spike_count'
    assert result.global_result == {0: 8, 1: 4, 2: 4}
    assert result.per_layer_result == [{0: 2, 1: 1, 2: 1},
                                       {0: 2, 1: 1, 2: 1},
                                       {0: 2, 1: 1, 2: 1},
                                       {0: 2, 1: 1, 2: 1}]


def test_changing_density_non_uniform_layers(non_uniform_layers):
    metric = spike_count.SpikeCount()

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
    assert result.metric_name == 'spike_count'
    assert result.global_result == {0: 9, 1: 5, 2: 6}
    assert result.per_layer_result == [
        [{0: 2, 1: 1, 2: 1},
         {0: 2, 1: 1, 2: 1},
         {0: 2, 1: 1, 2: 1},
         {0: 2, 1: 1, 2: 1}], {0: 1, 1: 1, 2: 2}]
