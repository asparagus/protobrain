#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from protobrain import neuron


@pytest.fixture(scope='module')
def num_neurons():
    return 10

@pytest.fixture(scope='module')
def expected_output(num_neurons):
    return np.random.rand(num_neurons) < 0.5

def test_null_inputs(num_neurons):
    neurons = neuron.Neurons(num_neurons)

    with pytest.raises(IndexError):
        neurons.input.values

def test_connect_input(num_neurons, expected_output):
    neurons = neuron.Neurons(num_neurons)
    other = neuron.Neurons(num_neurons)

    neurons.input = other
    other.output.values = expected_output

    assert np.all(neurons.input.values == expected_output)

def test_connect_additional_input(num_neurons, expected_output):
    neurons = neuron.Neurons(num_neurons)
    other = neuron.Neurons(num_neurons)

    neurons.set('feedback', other)
    other.output.values = expected_output

    assert np.all(neurons.get('feedback').values == expected_output)

def test_no_connections():
    layers = [neuron.Neurons(i) for i in [10, 10, 10]]

    for layer in layers:
        with pytest.raises(IndexError):
            layer.input.values

def test_feed_forward():
    layers = [neuron.Neurons(i) for i in [10, 10, 10]]
    neuron.FeedForward(layers)

    for i, layer in enumerate(layers[:-1]):
        layers[i + 1].input._connected_output == layer.output

def test_feed_forward_custom_input():
    layers = [neuron.Neurons(i) for i in [10, 10, 10]]
    neuron.FeedForward(layers, 'extra')

    for i, layer in enumerate(layers[:-1]):
        layers[i + 1].get('extra')._connected_output == layer.output

def test_feed_backward():
    layers = [neuron.Neurons(i) for i in [10, 10, 10]]
    neuron.FeedBackward(layers, 'feedback')

    for i, layer in enumerate(layers[:-1]):
        layer.get('feedback')._connected_output == layers[i + 1].output

def test_loop_back():
    layers = [neuron.Neurons(i) for i in [10, 10, 10]]
    neuron.LoopBack(layers, 'loopback')

    for layer in layers:
        layer.get('loopback')._connected_output == layer.output
