#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
from brain.event import EventVerifier
from brain.minicolumn import SimpleMiniColumnFactory


@pytest.fixture(scope='module')
def minicolumn_factory():
    return SimpleMiniColumnFactory(
        num_neurons=4,
        num_synapses=10,
        neuron_threshold=1e-10,
        synapse_threshold=1e-10
    )

def test_activation_triggers_spike(minicolumn_factory):
    minicol = minicolumn_factory.create()
    spike = EventVerifier(minicol.spike)

    minicol.active = True
    assert spike.has_run

def test_spike_propagation_when_connected(minicolumn_factory):
    minicol = minicolumn_factory.create()
    others = [minicolumn_factory.create()
              for _ in range(100)]

    minicol.connect(others)
    spike = EventVerifier(minicol.spike)

    for other in others:
        other.active = True

    # TODO: Ideally, wouldn't have to call this one.
    minicol.process()
    assert spike.has_run
