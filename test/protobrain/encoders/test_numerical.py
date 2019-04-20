#!/usr/bin/python
# -*- coding: utf-8 -*-
from protobrain.encoders import numerical


def test_fixed_range():
    fixed_range = numerical.SimpleEncoder(
        min_value=1, max_value=4, length=5, sparsity=0.4
    )
    assert all([1, 1, 0, 0, 0] == fixed_range.encode(1))
    assert all([0, 1, 1, 0, 0] == fixed_range.encode(2))
    assert all([0, 0, 1, 1, 0] == fixed_range.encode(3))
    assert all([0, 0, 0, 1, 1] == fixed_range.encode(4))

def test_cyclic():
    cyclic = numerical.CyclicEncoder(
        min_value=1, max_value=5, length=5, sparsity=0.4
    )
    assert all([1, 1, 0, 0, 0] == cyclic.encode(1))
    assert all([0, 1, 1, 0, 0] == cyclic.encode(2))
    assert all([0, 0, 1, 1, 0] == cyclic.encode(3))
    assert all([0, 0, 0, 1, 1] == cyclic.encode(4))
    assert all([1, 0, 0, 0, 1] == cyclic.encode(5))
