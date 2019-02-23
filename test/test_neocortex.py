#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from brain.event import EventVerifier
from brain.neocortex import SimpleNeocortexFactory


@pytest.fixture(scope='module')
def neocortex_factory():
    return SimpleNeocortexFactory(
        num_layers=10,
        num_minicolumns=100,
        num_neurons=4,
        num_synapses=10,
        synapse_threshold=1e-10
    )

def test_propagation(neocortex_factory):
    neocortex = neocortex_factory()
    first_layer = neocortex._layers[0]
    last_layer = neocortex._layers[-1]

    last_layer_spikes = [EventVerifier(minicolumn.spike)
                         for minicolumn in last_layer]

    # Set inputs at the first layer
    for minicolumn in first_layer:
        minicolumn._set_input(
            np.ones(minicolumn.num_synapses)
        )

    # Verify last layer spikes
    neocortex.process()
    assert all(last_layer_spikes)
