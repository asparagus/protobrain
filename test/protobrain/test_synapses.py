#!/usr/bin/python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from protobrain import synapses


@pytest.fixture(scope='module')
def shape():
    return 10

@pytest.fixture(scope='module')
def expected_output(shape):
    return np.random.rand(shape) < 0.5

def test_merge_output():
    out1 = synapses.Output(5)
    out2 = synapses.Output(8)

    outm = synapses.OutputMerge([out1, out2])

    assert outm.shape == (13,)

def test_slice_output():
    out1 = synapses.Output(5)
    outs = out1[3:]

    assert outs.shape == (2,)

def test_slice_output_2d():
    out1 = synapses.Output((10, 5))
    outs = out1[5:, 3:]

    assert outs.shape == (5, 2)

def test_output_validates_values():
    out = synapses.Output(10)

    with pytest.raises(ValueError):
        out.values = np.zeros(11)

def test_connect_input_output(shape, expected_output):
    inp = synapses.Input('', shape)
    out = synapses.Output(shape)

    inp.connect(out)
    out.values = expected_output
    assert all(inp.values == expected_output)

def test_connect_input_merged_output():
    inp = synapses.Input('', 10)
    out1 = synapses.Output(5)
    out2 = synapses.Output(5)
    out = synapses.OutputMerge([out1, out2])

    inp.connect(out)
    out1.values = np.ones(5)
    out2.values = np.ones(5) * 2

    assert all(inp.values == [1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2])

def test_connect_input_sliced_output(shape, expected_output):
    inp = synapses.Input('', 2)
    out = synapses.Output(shape)

    sliced = out[-2:]
    inp.connect(sliced)

    out.values = expected_output
    assert all(inp.values == expected_output[-2:])

def test_connect_2d():
    shape = (3, 4)
    expected_out = np.random.rand(*shape)
    inp = synapses.Input('', shape)
    out = synapses.Output(shape)

    inp.connect(out)
    out.values = expected_out
    for i in range(shape[0]):
        assert all(inp.values[i] == expected_out[i])
