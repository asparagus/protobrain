#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from brain import event
from brain import neocortex
from brain import synapses
from brain import synaptic_policy


def test_propagation(dummy_computation):
    num_layers = 100
    num_neurons = 100
    neocortex_factory = neocortex.SimpleNeocortexFactory(
        num_layers=num_layers,
        num_neurons=num_neurons,
        computation=dummy_computation(np.zeros(num_neurons)),
        synaptic_policies=[synaptic_policy.ForwardSynapticPolicy(synapses.InputConnection)],
        learning_function=None
    )

    cortex = neocortex_factory()
    last_layer = cortex._layers[-1]
    verify = event.EventVerifier(last_layer.emit)

    cortex.process()

    assert verify.has_run
