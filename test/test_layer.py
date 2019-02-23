#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
from brain.event import EventVerifier
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

def test_connection(layer_factory):
    layer1 = layer_factory.create()
    layer2 = layer_factory.create()

    layer2.connect(layer1)

    spikes = [EventVerifier(mc.spike)
              for mc in layer2._minicolumns]

    # Process does nothing while there's no activity in 1st layer
    layer2.process()
    assert not any(spike.has_run for spike in spikes)

    # Activating minicolumns in the 1st layer
    for mc in layer1._minicolumns:
        mc.active = 1

    # Now all minicolumns in the 2nd layer will have spiked
    layer2.process()
    assert all(spike.has_run for spike in spikes)
    assert all(spike.count == 1 for spike in spikes)
