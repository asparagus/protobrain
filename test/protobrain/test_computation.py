#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from protobrain import computation
from protobrain import neuron


def test_standard_computation():
    compute = computation.StandardComputation(threshold=1.5)

    values = np.array([1, 0, 1])
    strength = np.array([
        [1, 1, 1],  # 2.0
        [0, 1, 1],  # 1.0
        [0, 0, 1],  # 1.0
        [1, 0, 1]   # 2.0
    ])

    output = compute(neuron.NeuronInput(values, strength))
    expected = [True, False, False, True]

    assert np.all(output == expected)
