#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from brain.event import EventVerifier
from brain.neurons import SimpleNeuronsFactory


@pytest.fixture(scope='module')
def num_neurons():
    return 10

@pytest.fixture(scope='module')
def expected_output(num_neurons):
    return np.random.rand(num_neurons) < 0.5

@pytest.fixture(scope='module')
def neuron_factory(num_neurons, expected_output, dummy_computation):
    return SimpleNeuronsFactory(num_neurons, dummy_computation(expected_output))

def test_null_inputs(neuron_factory):
    neurons = neuron_factory()

    assert neurons._inputs is None
    assert neurons._feedback is None
    assert neurons._inhibitions is None

def test_length(num_neurons, neuron_factory):
    neurons = neuron_factory()
    assert len(neurons) == num_neurons

def test_emit(neuron_factory, expected_output):
    neurons = neuron_factory()
    verify = EventVerifier(neurons.emit)

    neurons.compute()
    assert verify.has_run
    assert verify.count == 1
    assert np.all(verify.run_args[0] == expected_output)
