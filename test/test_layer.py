#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
from brain.layer import SimpleLayerFactory


@pytest.fixture(scope='module')
def layer_factory():
    return SimpleLayerFactory(
        num_minicolumns=10,
        num_neurons=4,
        num_synapses=5,
        neuron_threshold=1e-10,
        synapse_threshold=1e-10
    )

def test_factory(layer_factory):
    layer = layer_factory()
