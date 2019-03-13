#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from brain.computation import StandardComputation
from brain.neurons import NeuronInput


def test_standard_computation():
    compute = StandardComputation(threshold=1.5)

    values = np.array([1, 0, 1])
    strength = np.array([
        [1, 1, 1],  # 2.0
        [0, 1, 1],  # 1.0
        [0, 0, 1],  # 1.0
        [1, 0, 1]   # 2.0
    ])

    output = compute(NeuronInput(values, strength))
    expected = [True, False, False, True]

    assert np.all(output == expected)
