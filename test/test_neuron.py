#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from brain.neuron import SimpleNeuronFactory


@pytest.fixture(scope='module')
def neuron_factory():
    return SimpleNeuronFactory(100, synapse_threshold=1e-10)

def test_output_true(neuron_factory):
    neuron = neuron_factory.create()
    _input = np.ones(neuron.num_synapses)
    _expected = True
    assert neuron.compute(_input) == _expected

def test_output_false(neuron_factory):
    neuron = neuron_factory.create()
    _input = np.zeros(neuron.num_synapses)
    _expected = False
    assert neuron.compute(_input) == _expected
