"""Tests for learning module."""

import numpy as np
from numpy import testing

from protobrain import learning
from protobrain import neuron


def test_hebbian_learning():
    hl = learning.HebbianLearning(increase=0.1, decrease=0.05)

    n1 = neuron.Neurons(3)
    n2 = neuron.Neurons(2, learning=hl)

    synapses = np.array([[0.8, 0.4], [0.6, 0.8], [0.5, 0.4]])

    n2.set(
        neuron.Neurons.MAIN_INPUT,
        n1,
        synapse_function=lambda inp_shape, out_shape: synapses,
    )

    n1.output.values = np.array([1, 0, 1])
    n2.output.values = np.array([1, 0])

    n2.learn()

    expected_synapses = np.array([[0.90, 0.35], [0.55, 0.80], [0.60, 0.35]])

    testing.assert_allclose(n2.input.synapses, expected_synapses)
