#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from brain.event import EventVerifier
from brain.neurons import SimpleNeuronsFactory
from brain.synapses import InputConnection
from brain.synapses import FeedbackConnection
from brain.synapses import InhibitionConnection
from brain.synaptic_policy import BackwardSynapticPolicy
from brain.synaptic_policy import ForwardSynapticPolicy
from brain.synaptic_policy import NullSynapticPolicy
from brain.synaptic_policy import ReflexiveSynapticPolicy
from brain.synaptic_policy import SpecificSynapticPolicy


@pytest.fixture(scope='module')
def num_neurons():
    return 10

@pytest.fixture(scope='module')
def neuron_factory(num_neurons, dummy_computation):
    return SimpleNeuronsFactory(
        num_neurons,
        dummy_computation(np.zeros(num_neurons))
    )

@pytest.fixture
def layers(neuron_factory):
    return [neuron_factory() for _ in range(3)]

def test_null_synaptic_policy(layers):
    nsp = NullSynapticPolicy()
    connections = nsp.execute(layers)

    assert len(connections) == 0

def test_forward_synaptic_policy(layers):
    fsp = ForwardSynapticPolicy(InputConnection)
    connections = fsp.execute(layers)

    assert len(connections) == len(layers) - 1
    for i in range(len(connections)):
        assert connections[i]._neurons == layers[i]
        assert connections[i]._other_neurons == layers[i + 1]

    for layer in layers:
        layer.compute()

    assert layers[0]._inputs is None
    for layer in layers[1:]:
        assert layers[i]._inputs is not None

def test_backward_synaptic_policy(layers):
    bsp = BackwardSynapticPolicy(InputConnection)
    connections = bsp.execute(layers)

    assert len(connections) == len(layers) - 1
    for i in range(len(connections)):
        assert connections[i]._neurons == layers[i + 1]
        assert connections[i]._other_neurons == layers[i]

    for layer in layers:
        layer.compute()

    assert layers[-1]._inputs is None
    for layer in layers[:-1]:
        assert layers[i]._inputs is not None

def test_reflexive_synaptic_policy(layers):
    rsp = ReflexiveSynapticPolicy(InputConnection)

    connections = rsp.execute(layers)

    assert len(connections) == len(layers)
    for i in range(len(connections)):
        assert connections[i]._neurons == layers[i]
        assert connections[i]._other_neurons == layers[i]

    for layer in layers:
        layer.compute()

    for layer in layers:
        assert layers[i]._inputs is not None

def test_specific_synaptic_policy(layers):
    ssp = SpecificSynapticPolicy(InputConnection, {2: 1})
    connections = ssp.execute(layers)

    assert len(connections) == 1
    assert connections[0]._neurons == layers[2]
    assert connections[0]._other_neurons == layers[1]

    for layer in layers:
        layer.compute()

    for layer in layers:
        assert layers[1]._inputs is not None
