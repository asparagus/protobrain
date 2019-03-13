#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from brain.event import EventVerifier
from brain.neocortex import SimpleNeocortexFactory
from brain.synapses import InputConnection
from brain.synaptic_policy import ForwardSynapticPolicy
from brain.synaptic_policy import NullSynapticPolicy


def test_propagation(dummy_computation):
    num_layers = 100
    num_neurons = 100
    factory = SimpleNeocortexFactory(
        num_layers=num_layers,
        num_neurons=num_neurons,
        computation=dummy_computation(np.zeros(num_neurons)),
        synaptic_policies=[ForwardSynapticPolicy(InputConnection)],
        learning_function=None
    )

    neocortex = factory()
    last_layer = neocortex._layers[-1]
    verify = EventVerifier(last_layer.emit)

    neocortex.process()

    assert verify.has_run
