"""Tests for computation module."""

import numpy as np

from protobrain import computation
from protobrain import neuron


def test_standard_computation():
    compute = computation.StandardComputation(threshold=1.2)

    n1 = neuron.Neurons(3)
    n2 = neuron.Neurons(2, computation=compute)

    synapses = np.array([[0.8, 0.4], [0.6, 0.8], [0.5, 0.4]])

    n2.set(
        neuron.Neurons.MAIN_INPUT,
        n1,
        synapse_function=lambda inp_shape, out_shape: synapses,
    )

    n1.output.values = np.array([1, 0, 1])
    output = n2.compute()

    expected = [True, False]

    assert all(output == expected)


def test_sparse_computation_with_fixed_number():
    compute = computation.SparseComputation(n=2)

    n1 = neuron.Neurons(3)
    n2 = neuron.Neurons(4, computation=compute)

    synapses = np.array(
        [[0.8, 0.4, 0.5, 0.4], [0.6, 0.8, 0.9, 0.2], [0.5, 0.4, 0.5, 0.7]]
    )

    n2.set(
        neuron.Neurons.MAIN_INPUT,
        n1,
        synapse_function=lambda inp_shape, out_shape: synapses,
    )

    n1.output.values = np.array([1, 0, 1])
    output = n2.compute()

    expected = [True, False, False, True]

    assert all(output == expected)


def test_sparse_computation_with_fixed_number():
    compute = computation.SparseComputation(n=0.25)

    n1 = neuron.Neurons(3)
    n2 = neuron.Neurons(4, computation=compute)

    synapses = np.array(
        [[0.8, 0.4, 0.5, 0.4], [0.6, 0.8, 0.9, 0.2], [0.5, 0.4, 0.5, 0.7]]
    )

    n2.set(
        neuron.Neurons.MAIN_INPUT,
        n1,
        synapse_function=lambda inp_shape, out_shape: synapses,
    )

    n1.output.values = np.array([1, 0, 1])
    output = n2.compute()

    expected = [True, False, False, False]

    assert all(output == expected)
